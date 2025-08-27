import json
import os
from re import A
import ray
from datetime import datetime
import random
from typing import Dict, List, Tuple
from pathlib import Path
from polars import col as c
import tqdm
from data_exporter.dig_a_plan_to_expansion import (
    dig_a_plan_to_expansion,
    remove_switches_from_grid_data,
)
from data_schema import NodeEdgeModel
from pipelines.expansion.admm_helpers import ADMM, ADMMResult
from pipelines.expansion.api import run_sddp
from pipelines.expansion.ltscenarios import generate_long_term_scenarios
from pipelines.expansion.models.request import (
    AdditionalParams,
    BenderCut,
    BenderCuts,
    Cut,
    ExpansionRequest,
    OptimizationConfig,
    PlanningParams,
    RiskMeasureType,
)
from pipelines.expansion.models.response import ExpansionResponse
from pipelines.helpers.json_rw import save_obj_to_json, load_obj_from_json

SERVER_RAY_ADDRESS = os.getenv("SERVER_RAY_ADDRESS", None)


def init_ray():
    ray.init(
        address=SERVER_RAY_ADDRESS,
        runtime_env={
            "working_dir": os.path.dirname(os.path.abspath(__file__)),
        },
    )
    return {
        "message": "Ray initialized",
        "nodes": ray.nodes(),
        "available_resources": ray.cluster_resources(),
        "used_resources": ray.available_resources(),
    }


def shutdown_ray():
    ray.shutdown()
    return {"message": "Ray shutdown"}


class ExpansionAlgorithm:

    def __init__(
        self,
        grid_data: NodeEdgeModel,
        each_task_memory: float,
        cache_dir: Path,
        admm_groups: int | Dict[int, List[int]] = 1,
        iterations: int = 10,
        n_admm_simulations: int = 10,
        seed_number: int = 42,
        time_limit: int = 10,
        solver_non_convex: int = 2,
        n_stages: int = 3,
        initial_budget: float = 1e6,
        discount_rate: float = 0.05,
        γ_cuts: float = 0.0,
        years_per_stage: int = 1,
        sddp_iteration_limit: int = 10,
        sddp_simulations: int = 100,
        just_test: bool = False,
        risk_measure_type: RiskMeasureType = RiskMeasureType.EXPECTATION,
        risk_measure_param: float = 0.1,
        δ_load_var: float = 0.1,
        δ_pv_var: float = 0.1,
        δ_b_var: float = 0.1,
        number_of_sddp_scenarios: int = 100,
        expansion_transformer_cost_per_kw: float = 1e3,
        expansion_line_cost_per_km_kw: float = 1e3,
        penalty_cost_per_consumption_kw: float = 1e3,
        penalty_cost_per_production_kw: float = 1e3,
        penalty_cost_per_infeasibility_kw: float = 1e3,
        s_base: float = 1e6,
        admm_big_m: float = 1e3,
        admm_ε: float = 1e-4,
        admm_ρ: float = 2.0,
        admm_γ_infeasibility: float = 1.0,
        admm_γ_admm_penalty: float = 1.0,
        admm_γ_trafo_loss: float = 1e2,
        admm_max_iters: int = 10,
        admm_μ: float = 10.0,
        admm_τ_incr: float = 2.0,
        admm_τ_decr: float = 2.0,
        with_ray: bool = False,
    ):
        self.grid_data = grid_data
        self.cache_dir = cache_dir
        self.each_task_memory = each_task_memory
        self.admm_groups = admm_groups
        self.iterations = iterations
        self.just_test = just_test
        self.n_admm_simulations = n_admm_simulations
        self.seed_number = seed_number
        self.time_limit = time_limit
        self.solver_non_convex = solver_non_convex
        self.with_ray = with_ray
        self.expansion_transformer_cost_per_kw = expansion_transformer_cost_per_kw
        self.expansion_line_cost_per_km_kw = expansion_line_cost_per_km_kw
        self.penalty_cost_per_consumption_kw = penalty_cost_per_consumption_kw
        self.penalty_cost_per_production_kw = penalty_cost_per_production_kw
        self.penalty_cost_per_infeasibility_kw = penalty_cost_per_infeasibility_kw
        self.s_base = s_base
        self.admm_big_m = admm_big_m
        self.admm_ε = admm_ε
        self.admm_ρ = admm_ρ
        self.admm_γ_infeasibility = admm_γ_infeasibility
        self.admm_γ_admm_penalty = admm_γ_admm_penalty
        self.admm_γ_trafo_loss = admm_γ_trafo_loss
        self.admm_max_iters = admm_max_iters
        self.admm_μ = admm_μ
        self.admm_τ_incr = admm_τ_incr
        self.admm_τ_decr = admm_τ_decr

        random.seed(seed_number)
        self.grid_data_rm = remove_switches_from_grid_data(self.grid_data)
        self.create_planning_params(
            n_stages=n_stages,
            initial_budget=initial_budget,
            discount_rate=discount_rate,
            γ_cuts=γ_cuts,
            years_per_stage=years_per_stage,
        )
        self.create_additional_params(
            iteration_limit=sddp_iteration_limit,
            n_simulations=sddp_simulations,
            risk_measure_type=risk_measure_type,
            risk_measure_param=risk_measure_param,
        )
        self.create_scenario_data(
            δ_load_var=δ_load_var,
            δ_pv_var=δ_pv_var,
            δ_b_var=δ_b_var,
            number_of_scenarios=number_of_sddp_scenarios,
            number_of_stages=n_stages,
        )
        self.create_bender_cuts()
        self.cache_dir_run = (
            self.cache_dir
            / "algorithm"
            / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(self.cache_dir_run, exist_ok=True)

    def _range(self, i: int):
        return range(1, 2 if self.just_test else i + 1)

    def create_planning_params(
        self,
        n_stages: int = 3,
        initial_budget: float = 1e6,
        discount_rate: float = 0.05,
        γ_cuts: float = 0.0,
        years_per_stage: int = 1,
    ):
        """Create planning parameters with default or custom values."""
        self.planning_params = PlanningParams(
            n_stages=n_stages,
            initial_budget=initial_budget,
            γ_cuts=γ_cuts,
            discount_rate=discount_rate,
            years_per_stage=years_per_stage,
            n_cut_scenarios=len(list(self.grid_data.load_data.keys())),
        )

    def create_scenario_data(
        self,
        δ_load_var=0.1,
        δ_pv_var=0.1,
        δ_b_var=0.1,
        number_of_scenarios=10,
        number_of_stages=3,
    ):
        """Generate long-term scenarios with configurable parameters."""
        self.scenario_data = generate_long_term_scenarios(
            nodes=self.grid_data_rm.node_data,
            δ_load_var=δ_load_var,
            δ_pv_var=δ_pv_var,
            δ_b_var=δ_b_var,
            number_of_scenarios=number_of_scenarios,
            number_of_stages=number_of_stages,
            seed_number=self.seed_number,
        )

    def create_additional_params(
        self,
        iteration_limit=10,
        n_simulations=100,
        risk_measure_type=RiskMeasureType.EXPECTATION,
        risk_measure_param=0.1,
    ):
        """Create additional parameters with default or custom values."""
        self.additional_params = AdditionalParams(
            iteration_limit=iteration_limit,
            n_simulations=n_simulations,
            risk_measure_type=risk_measure_type,
            risk_measure_param=risk_measure_param,
            seed=self.seed_number,
        )

    def create_bender_cuts(self, cuts=None):
        """Create Bender cuts with default or custom values."""
        self.bender_cuts = BenderCuts(cuts=cuts or {})

    def create_expansion_request(self):
        """Create expansion request with provided or default parameters."""
        self.expansion_request = dig_a_plan_to_expansion(
            grid_data=self.grid_data_rm,
            s_base=self.s_base,
            planning_params=self.planning_params,
            additional_params=self.additional_params,
            expansion_line_cost_per_km_kw=self.expansion_line_cost_per_km_kw,
            expansion_transformer_cost_per_kw=self.expansion_transformer_cost_per_kw,
            penalty_cost_per_consumption_kw=self.penalty_cost_per_consumption_kw,
            penalty_cost_per_production_kw=self.penalty_cost_per_production_kw,
            penalty_cost_per_infeasibility_kw=self.penalty_cost_per_infeasibility_kw,
            scenarios_data=self.scenario_data,
            bender_cuts=self.bender_cuts,
            scenarios_cache=self.cache_dir_run / "scenarios.json",
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
            bender_cuts=final_cuts,
        )

    def run_sddp(self) -> ExpansionResponse:
        """Run the SDDP algorithm with the given expansion request."""
        return run_sddp(self.expansion_request, self.cache_dir_run)

    def run_admm(self, sddp_response: ExpansionResponse, ι: int) -> BenderCuts:
        """Run the ADMM algorithm with the given expansion request."""
        admm = ADMM(
            groups=self.admm_groups,
            grid_data=self.grid_data,
            solver_non_convex=self.solver_non_convex,
            time_limit=self.time_limit,
            big_m=self.admm_big_m,
            ε=self.admm_ε,
            ρ=self.admm_ρ,
            γ_infeasibility=self.admm_γ_infeasibility,
            γ_admm_penalty=self.admm_γ_admm_penalty,
            γ_trafo_loss=self.admm_γ_trafo_loss,
            max_iters=self.admm_max_iters,
            μ=self.admm_μ,
            τ_incr=self.admm_τ_incr,
            τ_decr=self.admm_τ_decr,
        )
        (self.cache_dir_run / "admm").mkdir(parents=True, exist_ok=True)
        if self.with_ray:
            init_ray()
            self.check_ray()
            heavy_task_remote = ray.remote(memory=self.each_task_memory)(heavy_task)
            sddp_response_ref = ray.put(sddp_response)
            admm_ref = ray.put(admm)
            node_ids_ref = ray.put(self.node_ids)
            edge_ids_ref = ray.put(self.edge_ids)
            futures = {
                (stage, ω): heavy_task_remote.remote(
                    self.n_admm_simulations,
                    sddp_response_ref,
                    admm_ref,
                    stage,
                    node_ids_ref,
                    edge_ids_ref,
                )
                for stage in self._range(self.n_stages)
                for ω in self._range(self.n_admm_simulations)
            }
            future_results = {
                (stage, ω): ray.get(futures[(stage, ω)])
                for stage in self._range(self.n_stages)
                for ω in self._range(self.n_admm_simulations)
            }
            bender_cuts = BenderCuts(
                cuts={
                    self._cut_number(ι, stage, ω): future_results[(stage, ω)][0]
                    for stage in self._range(self.n_stages)
                    for ω in self._range(self.n_admm_simulations)
                }
            )
            shutdown_ray()
        else:
            bender_cuts = BenderCuts(cuts={})
            future_results = {}
            for stage in tqdm.tqdm(self._range(self.n_stages), desc="stages"):
                for ω in tqdm.tqdm(
                    self._range(self.n_admm_simulations), desc="scenarios"
                ):
                    bender_cut, results = heavy_task(
                        n_simulations=self.n_admm_simulations,
                        sddp_response=sddp_response,
                        admm=admm,
                        stage=stage,
                        node_ids=self.node_ids,
                        edge_ids=self.edge_ids,
                    )
                    bender_cuts.cuts[self._cut_number(ι, stage, ω)] = bender_cut
                    future_results[(stage, ω)] = (bender_cuts, results)

        for stage in self._range(self.n_stages):
            for ω in self._range(self.n_admm_simulations):
                save_obj_to_json(
                    obj=future_results[(stage, ω)][1],
                    path_filename=self.cache_dir_run
                    / "admm"
                    / f"admm_result_{ι}_{stage}_{ω}.json",
                )

        return bender_cuts

    def _cut_number(self, ι: int, stage: int, ω: int) -> str:
        """Generate a cut number based on the iteration, stage, and scenario."""
        return f"{(ι - 1) * self.n_admm_simulations * self.n_stages + (stage - 1) * self.n_admm_simulations + ω}"

    def check_ray(self):
        """Check if Ray is available and initialized."""
        try:
            import ray

            RAY_AVAILABLE = True
        except ImportError:
            RAY_AVAILABLE = False

        if RAY_AVAILABLE and self.with_ray and ray.is_initialized():
            print("Running Pipeline with Ray")
        else:
            print("Running Pipeline without Ray")

    def run_pipeline(self) -> ExpansionResponse:
        """Run the entire expansion pipeline."""
        self.create_expansion_request()
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
        .agg(c("value").sum().alias("value"))
    )
    return {row["id"]: row["value"] for row in df.to_dicts()}


def _transform_admm_result_into_bender_cuts(admm_result: ADMMResult) -> BenderCut:
    """Transform ADMM results into Bender cuts."""
    return BenderCut(
        λ_load=_calculate_cuts(admm_result, ["installed_cons"]),
        λ_pv=_calculate_cuts(admm_result, ["installed_prod"]),
        λ_cap=_calculate_cuts(admm_result, ["current_limit", "current_limit_tr"]),
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


def heavy_task(
    n_simulations: int,
    sddp_response: ExpansionResponse,
    admm: ADMM,
    stage: int,
    node_ids: List[int],
    edge_ids: List[int],
) -> Tuple[BenderCut, Dict]:
    import os

    os.environ["GRB_LICENSE_FILE"] = os.environ["HOME"] + "/gurobi_license/gurobi.lic"

    rand_ω = random.randint(0, n_simulations - 1)
    admm.update_grid_data(
        δ_load=sddp_response.simulations[rand_ω][stage - 1].δ_load,
        δ_pv=sddp_response.simulations[rand_ω][stage - 1].δ_pv,
        node_ids=node_ids,
        δ_cap=sddp_response.simulations[rand_ω][stage - 1].δ_cap,
        edge_ids=edge_ids,
    )
    admm_results = admm.solve()
    bender_cut = _transform_admm_result_into_bender_cuts(admm_results)
    return bender_cut, admm_results.results
