import os
from pydantic import BaseModel, ConfigDict
import patito as pt
import ray
import random
from typing import Dict, List
from pathlib import Path
from polars import col as c
import tqdm
from api.ray_utils import init_ray, shutdown_ray, check_ray
from data_exporter.dap_to_expansion import (
    dig_a_plan_to_expansion,
    remove_switches_from_grid_data,
)
from data_model import NodeEdgeModel, EdgeData
from data_model.expansion import LongTermUncertainty, SDDPConfig
from data_model.reconfiguration import ADMMConfig
from pipeline_expansion.admm_helpers import ADMM, ADMMResult
from api.sddp import run_sddp, generate_scenarios
from data_model.sddp import (
    AdditionalParams,
    BenderCut,
    BenderCuts,
    Cut,
    Node,
    ExpansionRequest,
    LongTermScenarioRequest,
    OptimizationConfig,
    PlanningParams,
    ExpansionResponse,
    Simulation,
)
from helpers.json import save_obj_to_json, load_obj_from_json


class ExpansionAlgorithm:

    def __init__(
        self,
        grid_data: NodeEdgeModel,
        admm_config: ADMMConfig,
        sddp_config: SDDPConfig,
        long_term_uncertainty: LongTermUncertainty,
        each_task_memory: float,
        time_now: str,
        cache_dir: Path = Path(".cache"),
        bender_cuts: BenderCuts | None = None,
        iterations: int = 10,
        seed_number: int = 42,
        γ_cuts: float = 0.0,
        just_test: bool = False,
        s_base: float = 1e6,
        with_ray: bool = False,
    ):
        self.grid_data = grid_data
        self.admm_config = admm_config
        self.sddp_config = sddp_config
        self.long_term_uncertainty = long_term_uncertainty
        self.cache_dir = cache_dir
        self.each_task_memory = each_task_memory
        self.iterations = iterations
        self.just_test = just_test
        self.seed_number = seed_number
        self.with_ray = with_ray
        self.s_base = s_base

        random.seed(seed_number)
        self.grid_data_rm = remove_switches_from_grid_data(self.grid_data)
        self.create_planning_params(
            γ_cuts=γ_cuts, n_stages=self.long_term_uncertainty.n_stages
        )
        self.create_additional_params(sddp_config=sddp_config)
        nodes = [
            Node(id=node["node_id"])
            for node in grid_data.node_data.iter_rows(named=True)
        ]
        self.scenario_data = self.create_scenario_data(
            nodes=nodes,
            n_stages=self.long_term_uncertainty.n_stages,
            seed_number=self.seed_number,
        )
        self.out_of_sample_scenarios = self.create_scenario_data(
            nodes=nodes,
            n_stages=self.long_term_uncertainty.n_stages,
            seed_number=self.seed_number + 1000,
        )

        self.create_bender_cuts(bender_cuts=bender_cuts)
        self.cache_dir_run = self.cache_dir / "algorithm" / time_now
        os.makedirs(self.cache_dir_run, exist_ok=True)

    def _range(self, i: int):
        return range(1, 2 if self.just_test else i + 1)

    def create_planning_params(self, γ_cuts: float = 0.0, n_stages: int = 3):
        """Create planning parameters with default or custom values."""
        self.planning_params = PlanningParams(
            n_stages=n_stages,
            initial_budget=self.sddp_config.initial_budget,
            γ_cuts=γ_cuts,
            discount_rate=self.sddp_config.discount_rate,
            years_per_stage=self.sddp_config.years_per_stage,
            n_cut_scenarios=len(list(self.grid_data.load_data.keys())),
        )

    def create_scenario_data(
        self,
        nodes: List[Node],
        n_stages=3,
        seed_number=1000,
    ):
        """Generate long-term scenarios with configurable parameters."""
        ltm_scenarios = LongTermScenarioRequest(
            n_scenarios=self.long_term_uncertainty.number_of_scenarios,
            n_stages=n_stages,
            nodes=nodes,
            load_potential={
                node.id: self.long_term_uncertainty.δ_load_var for node in nodes
            },
            pv_potential={
                node.id: self.long_term_uncertainty.δ_pv_var for node in nodes
            },
            yearly_budget=self.long_term_uncertainty.δ_b_var,
            N_years_per_stage=self.planning_params.years_per_stage,
            seed_number=seed_number,
        )
        return generate_scenarios(ltm_scenarios)

    def create_additional_params(self, sddp_config: SDDPConfig):
        """Create additional parameters with default or custom values."""
        self.additional_params = AdditionalParams(
            iteration_limit=sddp_config.iterations,
            n_simulations=sddp_config.n_simulations,
            risk_measure_type=sddp_config.risk_measure_type,
            risk_measure_param=sddp_config.risk_measure_param,
            seed=self.seed_number,
        )

    def create_bender_cuts(self, bender_cuts: BenderCuts | None):
        """Create Bender cuts with default or custom values."""
        self.bender_cuts = BenderCuts(cuts={}) if bender_cuts is None else bender_cuts

    def create_expansion_request(self):
        """Create expansion request with provided or default parameters."""
        self.expansion_request = dig_a_plan_to_expansion(
            grid_data=self.grid_data_rm,
            s_base=self.s_base,
            planning_params=self.planning_params,
            additional_params=self.additional_params,
            expansion_line_cost_per_km_kw=self.sddp_config.expansion_line_cost_per_km_kw,
            expansion_transformer_cost_per_kw=self.sddp_config.expansion_transformer_cost_per_kw,
            penalty_cost_per_consumption_kw=self.sddp_config.penalty_cost_per_consumption_kw,
            penalty_cost_per_production_kw=self.sddp_config.penalty_cost_per_production_kw,
            scenarios_data=self.scenario_data,
            out_of_sample_scenarios=self.out_of_sample_scenarios,
            bender_cuts=self.bender_cuts,
            scenarios_cache=self.cache_dir_run / "scenarios.json",
            out_of_sample_scenarios_cache=self.cache_dir_run
            / "out_of_sample_scenarios.json",
            bender_cuts_cache=self.cache_dir_run / "bender_cuts.json",
            optimization_config_cache=self.cache_dir_run / "optimization_config.json",
        )
        self.node_ids = [
            node.id for node in self.expansion_request.optimization.grid.nodes
        ]
        self.edge_ids = [
            edge.id for edge in self.expansion_request.optimization.grid.edges
        ]
        self.n_scenarios = len(self.expansion_request.scenarios.model_dump().keys())
        self.n_simulations = self.additional_params.n_simulations
        self.n_stages = self.expansion_request.optimization.planning_params.n_stages

    def record_update_cache(
        self,
        sddp_response: ExpansionResponse,
        admm_response: BenderCuts,
        ι: int,
    ):
        """Update Bender cuts with new values."""
        if self.just_test:
            return None
        save_obj_to_json(sddp_response, self.cache_dir_run / f"sddp_response_{ι}.json")
        save_obj_to_json(admm_response, self.cache_dir_run / f"bender_cuts_{ι}.json")
        current_cuts = BenderCuts(
            **load_obj_from_json(self.cache_dir_run / f"bender_cuts.json")
        )
        final_cuts = BenderCuts(cuts={**current_cuts.cuts, **admm_response.cuts})
        save_obj_to_json(final_cuts, self.cache_dir_run / f"bender_cuts.json")
        opt_config = OptimizationConfig(
            **load_obj_from_json(self.cache_dir_run / "optimization_config.json")
        )
        opt_config.grid.cuts = opt_config.grid.cuts + [
            Cut(id=int(cut_id)) for cut_id in admm_response.cuts.keys()
        ]
        save_obj_to_json(opt_config, self.cache_dir_run / "optimization_config.json")
        self.expansion_request = ExpansionRequest(
            optimization=opt_config,
            scenarios=self.expansion_request.scenarios,
            out_of_sample_scenarios=self.expansion_request.out_of_sample_scenarios,
            bender_cuts=final_cuts,
        )

    def run_sddp(self) -> ExpansionResponse:
        """Run the SDDP algorithm with the given expansion request."""
        return run_sddp(self.expansion_request, self.cache_dir_run)

    def run_admm(self, sddp_response: ExpansionResponse, ι: int) -> BenderCuts:
        """Run the ADMM algorithm with the given expansion request."""
        admm = ADMM(grid_data=self.grid_data, konfig=self.admm_config)
        (self.cache_dir_run / "admm").mkdir(parents=True, exist_ok=True)
        if self.with_ray:
            init_ray()
            check_ray(self.with_ray)
            heavy_task_remote = ray.remote(memory=self.each_task_memory)(heavy_task)
            admm_ref = ray.put(admm)
            node_ids_ref = ray.put(self.node_ids)
            edge_ids_ref = ray.put(self.edge_ids)
            futures = {
                self._cut_number(ι, stage, ω): heavy_task_remote.remote(
                    sddp_response.simulations[
                        random.randint(0, self.sddp_config.n_optimizations - 1)
                    ][stage - 1],
                    admm_ref,
                    node_ids_ref,
                    edge_ids_ref,
                )
                for stage in self._range(self.n_stages)
                for ω in self._range(self.sddp_config.n_optimizations)
            }
            future_results = {
                (ω, stage): ray.get(futures[self._cut_number(ι, stage, ω)])
                for stage in self._range(self.n_stages)
                for ω in self._range(self.sddp_config.n_optimizations)
            }
            for (ω, stage), result in future_results.items():
                print(result)
                print(f"ADMM Result (ω={ω}, stage={stage}): {result.admm_results}")
                print(f"Bender Cut (ω={ω}, stage={stage}): {result.bender_cut}")
            bender_cuts = BenderCuts(
                cuts={
                    self._cut_number(ι, stage, ω): future_results[(ω, stage)].bender_cut
                    for stage in self._range(self.n_stages)
                    for ω in self._range(self.sddp_config.n_optimizations)
                }
            )
            admm_results = {
                self._cut_number(ι, stage, ω): future_results[(ω, stage)].admm_results
                for stage in self._range(self.n_stages)
                for ω in self._range(self.sddp_config.n_optimizations)
            }
            shutdown_ray()
        else:
            bender_cuts = BenderCuts(cuts={})
            admm_results = {}
            for stage in tqdm.tqdm(self._range(self.n_stages), desc="stages"):
                for ω in tqdm.tqdm(
                    self._range(self.sddp_config.n_optimizations), desc="scenarios"
                ):
                    rand_ω = random.randint(0, self.sddp_config.n_optimizations - 1)
                    sddp_simulation = sddp_response.simulations[rand_ω][stage - 1]
                    heavy_task_output = heavy_task(
                        sddp_simulation=sddp_simulation,
                        admm=admm,
                        node_ids=self.node_ids,
                        edge_ids=self.edge_ids,
                    )
                    bender_cuts.cuts[self._cut_number(ι, stage, ω)] = (
                        heavy_task_output.bender_cut
                    )
                    admm_results[self._cut_number(ι, stage, ω)] = (
                        heavy_task_output.admm_results
                    )

        for stage in self._range(self.n_stages):
            for ω in self._range(self.sddp_config.n_optimizations):
                save_obj_to_json(
                    obj=admm_results[self._cut_number(ι, stage, ω)],
                    path_filename=self.cache_dir_run
                    / "admm"
                    / f"admm_result_iter{ι}_stage{stage}_scen{ω}.json",
                )

        return bender_cuts

    def _cut_number(self, ι: int, stage: int, ω: int) -> str:
        """Generate a cut number based on the iteration, stage, and scenario."""
        return f"{(ι - 1) * self.sddp_config.n_optimizations * self.n_stages + (stage - 1) * self.sddp_config.n_optimizations + ω}"

    def run_pipeline(self) -> ExpansionResponse:
        """Run the entire expansion pipeline."""
        self.create_expansion_request()
        ι = 0
        for ι in tqdm.tqdm(self._range(self.iterations), desc="Pipeline iteration"):
            sddp_response = self.run_sddp()
            admm_response = self.run_admm(sddp_response=sddp_response, ι=ι)
            self.record_update_cache(
                sddp_response=sddp_response, admm_response=admm_response, ι=ι
            )
        sddp_response = self.run_sddp()
        save_obj_to_json(
            sddp_response, self.cache_dir_run / f"sddp_response_{ι+1}.json"
        )
        return sddp_response


def _calculate_cuts(
    admm_result: ADMMResult, constraint_names: List[str]
) -> Dict[str, float]:
    df = (
        admm_result.duals.filter(c("name").is_in(constraint_names))[["id", "value"]]
        .group_by("id")
        .agg(c("value").abs().sum().alias("value"))
    )
    return {row["id"]: row["value"] for row in df.to_dicts()}


def _calculate_ftrs(
    λ_load: Dict[str, float],
    λ_pv: Dict[str, float],
    edges: pt.DataFrame[EdgeData],
) -> Dict[str, float]:
    return {
        str(row["edge_id"]): -(
            abs(
                λ_load.get(str(row["u_of_edge"]), 0)
                - λ_load.get(str(row["v_of_edge"]), 0)
            )
            + abs(
                λ_pv.get(str(row["v_of_edge"]), 0) - λ_pv.get(str(row["u_of_edge"]), 0)
            )
        )
        / 2
        for row in edges.iter_rows(named=True)
    }


def _transform_admm_result_into_bender_cuts(
    admm_result: ADMMResult, edges: pt.DataFrame[EdgeData]
) -> BenderCut:
    """Transform ADMM results into Bender cuts."""
    λ_load = _calculate_cuts(admm_result, ["installed_cons"])
    λ_pv = _calculate_cuts(admm_result, ["installed_prod"])
    λ_cap = _calculate_ftrs(λ_load, λ_pv, edges)
    return BenderCut(
        λ_load=λ_load,
        λ_pv=λ_pv,
        λ_cap=λ_cap,
        load0={
            str(row["node_id"]): row["cons_installed"]
            for row in admm_result.load0.to_dicts()
        },
        pv0={
            str(row["node_id"]): row["prod_installed"]
            for row in admm_result.pv0.to_dicts()
        },
        cap0={
            str(row["edge_id"]): row["p_max_pu"] for row in admm_result.cap0.to_dicts()
        },
        θ=sum(admm_result.θs["θ"]),
    )


class HeavyTaskOutput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    bender_cut: BenderCut
    admm_results: Dict


def heavy_task(
    sddp_simulation: Simulation,
    admm: ADMM,
    node_ids: List[int],
    edge_ids: List[int],
) -> HeavyTaskOutput:
    import os

    os.environ["GRB_LICENSE_FILE"] = os.environ["HOME"] + "/gurobi_license/gurobi.lic"
    admm.update_grid_data(
        δ_load=sddp_simulation.δ_load,
        δ_pv=sddp_simulation.δ_pv,
        node_ids=node_ids,
        δ_cap=sddp_simulation.δ_cap,
        edge_ids=edge_ids,
    )
    admm_results = admm.solve()
    edges = admm.grid_data.edge_data
    bender_cut = _transform_admm_result_into_bender_cuts(admm_results, edges)

    return HeavyTaskOutput(bender_cut=bender_cut, admm_results=admm_results.results)
