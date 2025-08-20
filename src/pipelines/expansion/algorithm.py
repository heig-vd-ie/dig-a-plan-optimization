import json
import os
from datetime import datetime
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
    PlanningParams,
    RiskMeasureType,
)
from pipelines.expansion.models.response import ExpansionResponse


class ExpansionAlgorithm:

    def __init__(
        self,
        grid_data: NodeEdgeModel,
        cache_dir: Path,
        admm_groups: int | Dict[int, List[int]] = 1,
        iterations: int = 10,
        just_test: bool = False,
    ):
        self.grid_data = grid_data
        self.cache_dir = cache_dir
        self.admm_groups = admm_groups
        self.iterations = iterations
        self.just_test = just_test
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
        self, n_stages=3, initial_budget=100000, discount_rate=0.05, γ_cuts=0.0
    ):
        """Create planning parameters with default or custom values."""
        self.planning_params = PlanningParams(
            n_stages=n_stages,
            initial_budget=initial_budget,
            γ_cuts=γ_cuts,
            discount_rate=discount_rate,
        )

    def create_scenario_data(
        self,
        δ_load_var=0.1,
        δ_pv_var=0.1,
        δ_b_var=0.1,
        number_of_scenarios=10,
        number_of_stages=3,
        seed_number=42,
    ):
        """Generate long-term scenarios with configurable parameters."""
        self.scenario_data = generate_long_term_scenarios(
            nodes=self.grid_data_rm.node_data,
            δ_load_var=δ_load_var,
            δ_pv_var=δ_pv_var,
            δ_b_var=δ_b_var,
            number_of_scenarios=number_of_scenarios,
            number_of_stages=number_of_stages,
            seed_number=seed_number,
        )

    def create_additional_params(
        self,
        iteration_limit=10,
        n_simulations=100,
        risk_measure_type=RiskMeasureType.EXPECTATION,
        risk_measure_param=0.1,
        seed=42,
    ):
        """Create additional parameters with default or custom values."""
        self.additional_params = AdditionalParams(
            iteration_limit=iteration_limit,
            n_simulations=n_simulations,
            risk_measure_type=risk_measure_type,
            risk_measure_param=risk_measure_param,
            seed=seed,
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
        self.n_stages = self.expansion_request.optimization.planning_params.n_stages

    @staticmethod
    def _calculate_cuts(
        admm_result: ADMMResult, constraint_names: List[str]
    ) -> Dict[str, float]:
        df = (
            admm_result.duals.filter(c("name").is_in(constraint_names))[["id", "value"]]
            .group_by("id")
            .agg(c("value").sum().alias("value"))
        )
        return {row["id"]: row["value"] for row in df.to_dicts()}

    def transform_admm_result_into_bender_cuts(
        self, admm_result: ADMMResult
    ) -> BenderCut:
        """Transform ADMM results into Bender cuts."""
        return BenderCut(
            λ_load=self._calculate_cuts(admm_result, ["installed_cons"]),
            λ_pv=self._calculate_cuts(admm_result, ["installed_prod"]),
            λ_cap=self._calculate_cuts(
                admm_result, ["current_limit", "current_limit_tr"]
            ),
            load0={
                str(row["node_id"]): row["cons_installed"]
                for row in admm_result.load0.to_dicts()
            },
            pv0={
                str(row["node_id"]): row["prod_installed"]
                for row in admm_result.pv0.to_dicts()
            },
            cap0={
                str(row["edge_id"]): row["p_max_pu"]
                for row in admm_result.cap0.to_dicts()
            },
            θ=sum(admm_result.θs["θ"]),
        )

    def record_update_cache(
        self,
        sddp_response: ExpansionResponse,
        admm_response: BenderCuts,
        ι: int,
    ):
        """Update Bender cuts with new values."""
        if not self.just_test:
            sddp_response_dict = sddp_response.model_dump()
            with open(self.cache_dir_run / f"sddp_response_{ι}.json", "w") as f:
                json.dump(sddp_response_dict, f)
            bender_cuts_dict = admm_response.model_dump()
            with open(self.cache_dir_run / f"bender_cuts_{ι}.json", "w") as f:
                json.dump(bender_cuts_dict, f)
            with open(self.cache_dir_run / f"bender_cuts.json", "r") as f:
                current_bender_cuts = json.load(f)
            final_bender_cuts = {**current_bender_cuts, **bender_cuts_dict}
            with open(self.cache_dir_run / f"bender_cuts.json", "w") as f:
                json.dump(final_bender_cuts, f)

    def run_sddp(self) -> ExpansionResponse:
        """Run the SDDP algorithm with the given expansion request."""
        return run_sddp(self.expansion_request, self.cache_dir_run)

    def run_admm(self, sddp_response: ExpansionResponse, ι: int) -> BenderCuts:
        """Run the ADMM algorithm with the given expansion request."""
        admm = ADMM(groups=self.admm_groups, grid_data=self.grid_data)
        bender_cuts = BenderCuts(cuts={})
        for stage in tqdm.tqdm(self._range(self.n_stages), desc="stages"):
            for ω in tqdm.tqdm(self._range(self.n_scenarios), desc="scenarios"):
                admm.update_grid_data(
                    δ_load=sddp_response.simulations[ω][stage].δ_load,
                    δ_pv=sddp_response.simulations[ω][stage].δ_pv,
                    node_ids=self.node_ids,
                    δ_cap=sddp_response.simulations[ω][stage].δ_cap,
                    edge_ids=self.edge_ids,
                )
                admm_results = admm.solve()
                bender_cut = self.transform_admm_result_into_bender_cuts(admm_results)
                bender_cuts.cuts[f"{ι}_{ω}_{stage}"] = bender_cut
        return bender_cuts

    def run_pipeline(self):
        """Run the entire expansion pipeline."""
        self.create_expansion_request()
        for ι in tqdm.tqdm(self._range(self.iterations), desc="Pipeline iteration"):
            sddp_response = self.run_sddp()
            admm_response = self.run_admm(sddp_response=sddp_response, ι=ι)
            self.record_update_cache(
                sddp_response=sddp_response, admm_response=admm_response, ι=ι
            )
        return None
