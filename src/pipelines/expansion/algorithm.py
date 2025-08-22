import os
import ray
from datetime import datetime
import random
from typing import Dict, List
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


class ExpansionAlgorithm:

    def __init__(
        self,
        grid_data: NodeEdgeModel,
        cache_dir: Path,
        admm_groups: int | Dict[int, List[int]] = 1,
        iterations: int = 10,
        n_admm_simulations: int = 10,
        seed_number: int = 42,
        time_limit: int = 10,
        solver_non_convex: int = 2,
        just_test: bool = False,
        with_ray: bool = False,
    ):
        self.grid_data = grid_data
        self.cache_dir = cache_dir
        self.admm_groups = admm_groups
        self.iterations = iterations
        self.just_test = just_test
        self.n_admm_simulations = n_admm_simulations
        self.seed_number = seed_number
        self.time_limit = time_limit
        self.solver_non_convex = solver_non_convex
        self.with_ray = with_ray
        random.seed(seed_number)
        self.grid_data_rm = remove_switches_from_grid_data(self.grid_data)
        self.create_planning_params()
        self.create_additional_params()
        self.create_scenario_data()
        self.create_bender_cuts()
        self.cache_dir_run = (
            self.cache_dir
            / "algorithm"
            / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(self.cache_dir_run, exist_ok=True)

    def _range(self, i: int):
        return range(1 if self.just_test else i)

    def create_planning_params(
        self,
        n_stages=3,
        initial_budget=100000,
        discount_rate=0.05,
        γ_cuts=0.0,
        years_per_stage=1,
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
            planning_params=self.planning_params,
            additional_params=self.additional_params,
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
        )
        if self.run_by_ray:
            heavy_task_remote = ray.remote(heavy_task)
            futures = {
                (stage, ω): heavy_task_remote.remote(
                    self.n_admm_simulations,
                    sddp_response,
                    admm,
                    stage,
                    self.node_ids,
                    self.edge_ids,
                )
                for stage in self._range(self.n_stages)
                for ω in self._range(self.n_admm_simulations)
            }
            bender_cuts = BenderCuts(
                cuts={
                    self._cut_number(ι, stage, ω): ray.get(futures[(stage, ω)])
                    for stage in self._range(self.n_stages)
                    for ω in self._range(self.n_admm_simulations)
                }
            )
        else:
            bender_cuts = BenderCuts(cuts={})
            for stage in tqdm.tqdm(self._range(self.n_stages), desc="stages"):
                for ω in tqdm.tqdm(
                    self._range(self.n_admm_simulations), desc="scenarios"
                ):
                    bender_cut = heavy_task(
                        n_simulations=self.n_admm_simulations,
                        sddp_response=sddp_response,
                        admm=admm,
                        stage=stage,
                        node_ids=self.node_ids,
                        edge_ids=self.edge_ids,
                    )
                    bender_cuts.cuts[self._cut_number(ι, stage, ω)] = bender_cut
        return bender_cuts

    def _cut_number(self, ι: int, stage: int, ω: int) -> str:
        """Generate a cut number based on the iteration, stage, and scenario."""
        return f"{(ι - 1) * self.iterations + (stage - 1) * self.n_stages + ω}"

    def check_ray(self):
        """Check if Ray is available and initialized."""
        try:
            import ray

            RAY_AVAILABLE = True
        except ImportError:
            RAY_AVAILABLE = False

        if RAY_AVAILABLE and self.with_ray and ray.is_initialized():
            print("Running Pipeline with Ray")
            self.run_by_ray = True
        else:
            self.run_by_ray = False
            print("Running Pipeline without Ray")

    def run_pipeline(self):
        """Run the entire expansion pipeline."""
        self.check_ray()
        self.create_expansion_request()
        for ι in tqdm.tqdm(self._range(self.iterations), desc="Pipeline iteration"):
            sddp_response = self.run_sddp()
            admm_response = self.run_admm(sddp_response=sddp_response, ι=ι)
            self.record_update_cache(
                sddp_response=sddp_response, admm_response=admm_response, ι=ι
            )
        return None


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
):
    import os

    os.environ["GRB_LICENSE_FILE"] = os.environ["HOME"] + "/gurobi_license/gurobi.lic"

    rand_ω = random.randint(0, n_simulations)
    admm.update_grid_data(
        δ_load=sddp_response.simulations[rand_ω][stage].δ_load,
        δ_pv=sddp_response.simulations[rand_ω][stage].δ_pv,
        node_ids=node_ids,
        δ_cap=sddp_response.simulations[rand_ω][stage].δ_cap,
        edge_ids=edge_ids,
    )
    admm_results = admm.solve()
    bender_cut = _transform_admm_result_into_bender_cuts(admm_results)
    return bender_cut
