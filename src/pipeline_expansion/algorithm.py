import os
import json
from pydantic import BaseModel, ConfigDict
import patito as pt
import polars as pl
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
from data_model.expansion import SDDPConfig
from data_model.reconfiguration import ADMMConfig
from pipeline_expansion.admm_helpers import ADMM, ADMMResult
from api.sddp import SddpModel
from data_model.sddp import (
    AdditionalParams,
    BenderCut,
    BenderCuts,
    Cut,
    HeavyTaskConfig,
    Node,
    SddpRequest,
    LongTermScenarioRequest,
    OptimizationConfig,
    PlanningParams,
    SddpResponse,
    Simulation,
)
from helpers.json import save_obj_to_json, load_obj_from_json
from konfig import PROJECT_ROOT
from helpers import generate_log

log = generate_log(name=__name__)


class ExpansionAlgorithm:

    def __init__(
        self,
        grid_data: NodeEdgeModel,
        admm_config: ADMMConfig,
        sddp_config: SDDPConfig,
        load_potential: Dict[int, float],
        pv_potential: Dict[int, float],
        time_now: str,
        cache_dir: Path,
        each_task_memory: float,
        fixed_switches: bool = False,
        bender_cuts: BenderCuts | None = None,
        iterations: int = 10,
        seed_number: int = 42,
        just_test: bool = False,
        s_base: float = 1e6,
        with_ray: bool = False,
    ):

        self.grid_data = grid_data
        self.admm_config = admm_config
        self.sddp_config = sddp_config
        self.cache_dir = cache_dir
        self.cache_dir_run = self.cache_dir / time_now
        self.iterations = iterations
        self.just_test = just_test
        self.seed_number = seed_number
        self.with_ray = with_ray
        self.s_base = s_base
        self.each_task_memory = each_task_memory
        self.fixed_switches = fixed_switches

        self.expansion_model = SddpModel()

        random.seed(seed_number)
        self.grid_data_rm = remove_switches_from_grid_data(self.grid_data)
        self.create_planning_params(n_stages=self.sddp_config.n_stages)
        self.create_additional_params(sddp_config=sddp_config)
        nodes = [
            Node(id=node["node_id"])
            for node in grid_data.node_data.iter_rows(named=True)
        ]
        self.scenario_data = self.create_scenario_data(
            nodes=nodes,
            load_potential=load_potential,
            pv_potential=pv_potential,
            n_stages=self.sddp_config.n_stages,
            seed_number=self.seed_number,
            file_name="scenarios.json",
        )
        self.out_of_sample_scenarios = self.create_scenario_data(
            nodes=nodes,
            load_potential=load_potential,
            pv_potential=pv_potential,
            n_stages=self.sddp_config.n_stages,
            seed_number=self.seed_number + 1000,
            file_name="out_of_sample_scenarios.json",
        )

        self.create_bender_cuts(bender_cuts=bender_cuts)
        os.makedirs(self.cache_dir_run, exist_ok=True)

    def _range(self, i: int):
        return range(1, 2 if self.just_test else i + 1)

    def create_planning_params(self, n_stages: int = 3):
        """Create planning parameters with default or custom values."""
        self.planning_params = PlanningParams(
            n_stages=n_stages,
            initial_budget=self.sddp_config.initial_budget,
            discount_rate=self.sddp_config.discount_rate,
            years_per_stage=self.sddp_config.years_per_stage,
            n_cut_scenarios=len(list(self.grid_data.load_data.keys())),
        )

    def create_scenario_data(
        self,
        nodes: List[Node],
        load_potential: dict[int, float],
        pv_potential: dict[int, float],
        file_name: str,
        n_stages: int,
        seed_number: int,
    ):
        """Generate long-term scenarios with configurable parameters."""
        ltm_scenarios = LongTermScenarioRequest(
            n_scenarios=self.sddp_config.n_scenarios,
            n_stages=n_stages,
            nodes=nodes,
            load_potential=load_potential,
            pv_potential=pv_potential,
            yearly_budget=self.sddp_config.δ_b_var,
            N_years_per_stage=self.planning_params.years_per_stage,
            seed_number=seed_number,
        )
        scenarios = self.expansion_model.run_generate_scenarios(ltm_scenarios)
        save_obj_to_json(scenarios, self.cache_dir_run / file_name)
        return scenarios

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
        save_obj_to_json(self.bender_cuts, self.cache_dir_run / "bender_cuts.json")

    def create_sddp_request(self):
        """Create expansion request with provided or default parameters."""
        self.sddp_request = dig_a_plan_to_expansion(
            grid_data=self.grid_data_rm,
            s_base=self.s_base,
            planning_params=self.planning_params,
            additional_params=self.additional_params,
            expansion_line_cost_per_km_kw=self.sddp_config.expansion_line_cost_per_km_kw,
            expansion_transformer_cost_per_kw=self.sddp_config.expansion_transformer_cost_per_kw,
            penalty_cost_per_consumption_kw=self.sddp_config.penalty_cost_per_consumption_kw,
            penalty_cost_per_production_kw=self.sddp_config.penalty_cost_per_production_kw,
            bender_cuts=self.bender_cuts,
            scenarios_cache=self.cache_dir_run / "scenarios.json",
            out_of_sample_scenarios_cache=self.cache_dir_run
            / "out_of_sample_scenarios.json",
            bender_cuts_cache=self.cache_dir_run / "bender_cuts.json",
            optimization_config_cache=self.cache_dir_run / "optimization_config.json",
        )
        self.node_ids = [node.id for node in self.sddp_request.optimization.grid.nodes]
        self.edge_ids = [edge.id for edge in self.sddp_request.optimization.grid.edges]

    def record_batch_sddp(self, sddp_response: SddpResponse, ι: int):
        if len(sddp_response.objectives) == 0:
            return
        single_row = SddpResponse(
            objectives=[sddp_response.objectives[0]],
            simulations=[sddp_response.simulations[0]],
            out_of_sample_simulations=[sddp_response.out_of_sample_simulations[0]],
            out_of_sample_objectives=[sddp_response.out_of_sample_objectives[0]],
        )

        size_bytes = len(json.dumps(single_row.model_dump(mode="json")).encode("utf-8"))
        max_memory_bytes = 20 * 1024 * 1024
        batch_size = max(1, max_memory_bytes // size_bytes)
        n = len(sddp_response.objectives)
        for i in range(0, n, batch_size):
            batch = SddpResponse(
                objectives=sddp_response.objectives[i : i + batch_size],
                simulations=sddp_response.simulations[i : i + batch_size],
                out_of_sample_simulations=sddp_response.out_of_sample_simulations[
                    i : i + batch_size
                ],
                out_of_sample_objectives=sddp_response.out_of_sample_objectives[
                    i : i + batch_size
                ],
            )
            save_obj_to_json(
                batch,
                self.cache_dir_run / f"sddp_response_{ι}_{i//batch_size}.json",
            )

    def record_update_cache(self, admm_response: BenderCuts, ι: int):
        """Update Bender cuts with new values."""
        if self.just_test:
            return None
        save_obj_to_json(admm_response, self.cache_dir_run / f"bender_cuts_{ι}.json")
        current_cuts = load_obj_from_json(self.cache_dir_run / f"bender_cuts.json")
        final_cuts = BenderCuts(cuts={**current_cuts["cuts"], **admm_response.cuts})
        save_obj_to_json(final_cuts, self.cache_dir_run / f"bender_cuts.json")
        opt_config = OptimizationConfig(
            **load_obj_from_json(self.cache_dir_run / "optimization_config.json")
        )
        opt_config.grid.cuts = opt_config.grid.cuts + [
            Cut(id=int(cut_id)) for cut_id in admm_response.cuts.keys()
        ]
        save_obj_to_json(opt_config, self.cache_dir_run / "optimization_config.json")
        self.sddp_request = SddpRequest(optimization=opt_config)

    def run_sddp(self) -> SddpResponse:
        """Run the SDDP algorithm with the given expansion request."""
        import resource, gc

        gc.collect()
        mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        request_size_mb = (
            len(self.sddp_request.model_dump_json().encode()) / 1024 / 1024
        )
        log.info(
            f"PRE-SDDP | process RSS: {mem_mb:.1f} MB | request payload: {request_size_mb:.1f} MB"
        )

        response = self.expansion_model.run_sddp(self.sddp_request)

        mem_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        response_size_mb = len(response.model_dump_json().encode()) / 1024 / 1024
        log.info(
            f"POST-SDDP | RSS: {mem_after:.1f} MB | response: {response_size_mb:.1f} MB | delta: {mem_after - mem_mb:.1f} MB"
        )

        return response

    def run_admm(
        self, needed_simulations: Dict, ι: int, fixed_switches: bool
    ) -> BenderCuts:
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
                _cut_number(
                    ι,
                    stage,
                    ω,
                    self.sddp_config.n_optimizations,
                    self.sddp_config.n_stages,
                ): heavy_task_remote.remote(
                    needed_simulations[(stage, ω)],
                    admm_ref,
                    node_ids_ref,
                    edge_ids_ref,
                    fixed_switches,
                    HeavyTaskConfig(
                        ι=ι,
                        stage=stage,
                        ω=ω,
                        cache_dir_run=str(self.cache_dir_run),
                    ),
                )
                for stage in self._range(self.sddp_config.n_stages)
                for ω in self._range(self.sddp_config.n_optimizations)
            }
            future_results = {}
            for stage in self._range(self.sddp_config.n_stages):
                for ω in self._range(self.sddp_config.n_optimizations):
                    try:
                        future_results[(ω, stage)] = ray.get(
                            futures[
                                _cut_number(
                                    ι,
                                    stage,
                                    ω,
                                    self.sddp_config.n_optimizations,
                                    self.sddp_config.n_stages,
                                )
                            ]
                        )
                    except Exception as e:
                        log.error(f"ERROR in [{ω}, {stage}]!!! Exception: {e}")
            cuts = {}
            for stage in self._range(self.sddp_config.n_stages):
                for ω in self._range(self.sddp_config.n_optimizations):
                    try:
                        cuts[
                            _cut_number(
                                ι,
                                stage,
                                ω,
                                self.sddp_config.n_optimizations,
                                self.sddp_config.n_stages,
                            )
                        ] = future_results[(ω, stage)].bender_cut
                    except Exception as e:
                        log.error(
                            f"ERROR in retrieving data for [{ω}, {stage}]!!! Exception: {e}"
                        )
            bender_cuts = BenderCuts(cuts=cuts)

            shutdown_ray(futures=list(futures.values()))
            del admm_ref
            del node_ids_ref
            del edge_ids_ref
            del futures
            del future_results
            del needed_simulations

        else:
            bender_cuts = BenderCuts(cuts={})
            for stage in tqdm.tqdm(
                self._range(self.sddp_config.n_stages), desc="stages"
            ):
                for ω in tqdm.tqdm(
                    self._range(self.sddp_config.n_optimizations), desc="scenarios"
                ):
                    sddp_simulation = needed_simulations[(stage, ω)]
                    try:
                        heavy_task_output = heavy_task(
                            sddp_simulation=sddp_simulation,
                            admm=admm,
                            node_ids=self.node_ids,
                            edge_ids=self.edge_ids,
                            fixed_switches=fixed_switches,
                            heavy_task_config=HeavyTaskConfig(
                                ι=ι,
                                stage=stage,
                                ω=ω,
                                cache_dir_run=str(self.cache_dir_run),
                            ),
                        )
                        bender_cuts.cuts[
                            _cut_number(
                                ι,
                                stage,
                                ω,
                                self.sddp_config.n_optimizations,
                                self.sddp_config.n_stages,
                            )
                        ] = heavy_task_output.bender_cut
                    except Exception as e:
                        log.error(
                            f"ERROR in [{ω}, {stage}] without ray!!! Exception: {e}"
                        )
        return bender_cuts

    def run_pipeline(self) -> SddpResponse:
        """Run the entire expansion pipeline."""
        self.create_sddp_request()
        ι = 0
        for ι in tqdm.tqdm(self._range(self.iterations), desc="Pipeline iteration"):
            sddp_response = self.run_sddp()
            needed_simulations = {
                (stage, ω): sddp_response.simulations[
                    random.randint(0, self.sddp_config.n_optimizations - 1)
                ][stage - 1]
                for stage in self._range(self.sddp_config.n_stages)
                for ω in self._range(self.sddp_config.n_optimizations)
            }
            self.record_batch_sddp(sddp_response=sddp_response, ι=ι)
            del sddp_response
            admm_response = self.run_admm(
                needed_simulations=needed_simulations,
                ι=ι,
                fixed_switches=self.fixed_switches,
            )
            self.record_update_cache(admm_response=admm_response, ι=ι)
        sddp_response = self.run_sddp()
        self.record_batch_sddp(sddp_response=sddp_response, ι=ι)

        return sddp_response


def _transform_admm_result_into_bender_cuts(
    admm_result: ADMMResult, edges: pt.DataFrame[EdgeData]
) -> BenderCut:
    """Transform ADMM results into Bender cuts."""

    ds_cons = admm_result.dps_cons.join(admm_result.dqs_cons, on="node_id", how="left")
    ds_prod = admm_result.dps_prod.join(admm_result.dqs_prod, on="node_id", how="left")

    ds_cons = ds_cons.select(
        [
            c("node_id"),
            pl.sum_horizontal([c("p_curt_cons"), c("q_curt_cons")]).alias("val"),
        ]
    )
    ds_prod = ds_prod.select(
        [
            c("node_id"),
            pl.sum_horizontal([c("p_curt_prod"), c("q_curt_prod")]).alias("val"),
        ]
    )
    dv_relx = admm_result.dvs_relx_up.join(
        admm_result.dvs_relx_down, on="node_id", how="left"
    )
    dv_relx = dv_relx.select(
        [
            c("node_id"),
            pl.sum_horizontal([c("v_relax_up"), c("v_relax_down")]).alias("val"),
        ]
    )

    return BenderCut(
        λ_load={str(row["node_id"]): row["val"] for row in ds_cons.to_dicts()},
        λ_pv={str(row["node_id"]): row["val"] for row in ds_prod.to_dicts()},
        λ_v={str(row["node_id"]): row["val"] for row in dv_relx.to_dicts()},
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
    )


class HeavyTaskOutput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    bender_cut: BenderCut


def heavy_task(
    sddp_simulation: Simulation,
    admm: ADMM,
    node_ids: List[int],
    edge_ids: List[int],
    fixed_switches: bool,
    heavy_task_config: HeavyTaskConfig,
) -> HeavyTaskOutput:
    import os

    ι, stage, ω, cache_dir_run = (
        heavy_task_config.ι,
        heavy_task_config.stage,
        heavy_task_config.ω,
        heavy_task_config.cache_dir_run,
    )
    os.environ["GRB_LICENSE_FILE"] = os.environ["HOME"] + "/gurobi_license/gurobi.lic"
    admm.update_grid_data(
        δ_load=sddp_simulation.δ_load,
        δ_pv=sddp_simulation.δ_pv,
        node_ids=node_ids,
        δ_cap=sddp_simulation.δ_cap,
        edge_ids=edge_ids,
    )
    recorded_edge_data = admm.grid_data.edge_data.to_dicts()
    with open(
        PROJECT_ROOT
        / cache_dir_run
        / "admm"
        / f"recorded_edge_data_iter{ι}_stage{stage}_scen{ω}.json",
        "w",
    ) as f:
        json.dump(recorded_edge_data, f, indent=2)

    admm_results = admm.solve(fixed_switches=fixed_switches)
    edges = admm.grid_data.edge_data
    bender_cut = _transform_admm_result_into_bender_cuts(admm_results, edges)

    save_obj_to_json(
        obj=admm_results.results,
        path_filename=PROJECT_ROOT
        / cache_dir_run
        / "admm"
        / f"admm_result_iter{ι}_stage{stage}_scen{ω}.json",
    )
    log.info(
        f"admm_result_iter{ι}_stage{stage}_scen{ω}.json is written in desired location {cache_dir_run}"
    )

    result_heavy_task = HeavyTaskOutput(bender_cut=bender_cut)
    return result_heavy_task


def _cut_number(ι: int, stage: int, ω: int, n_optimizations: int, n_stages: int) -> str:
    """Generate a cut number based on the iteration, stage, and scenario."""
    return f"{(ι - 1) * n_optimizations * n_stages + (stage - 1) * n_optimizations + ω}"
