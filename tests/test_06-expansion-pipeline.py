import math
import numpy as np
import pytest
from data_exporter.pandapower_to_dig_a_plan import pandapower_to_dig_a_plan_schema
from data_exporter.dig_a_plan_to_expansion import (
    dig_a_plan_to_expansion,
    remove_switches_from_grid_data,
)
from pipelines.expansion.admm_helpers import ADMM
from pipelines.expansion.models.request import (
    PlanningParams,
    AdditionalParams,
    BenderCuts,
    RiskMeasureType,
)
from pipelines.expansion.ltscenarios import generate_long_term_scenarios
from pipelines.expansion.api import run_sddp


class ExpansionTestBase:
    """Base class for expansion pipeline tests with common setup."""

    @pytest.fixture(autouse=True)
    def setup_common_data(
        self, test_simple_grid, test_cache_dir, test_simple_grid_groups
    ):
        """Set up common test data and configurations."""
        self.net = test_simple_grid
        self.grid_data = pandapower_to_dig_a_plan_schema(self.net)
        self.grid_data_rm = remove_switches_from_grid_data(self.grid_data)
        self.cache_dir = test_cache_dir
        self.simple_grid_groups = test_simple_grid_groups

    def create_planning_params(
        self, n_stages=3, initial_budget=100000, discount_rate=0.05
    ):
        """Create planning parameters with default or custom values."""
        return PlanningParams(
            n_stages=n_stages,
            initial_budget=initial_budget,
            discount_rate=discount_rate,
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
        return AdditionalParams(
            iteration_limit=iteration_limit,
            n_simulations=n_simulations,
            risk_measure_type=risk_measure_type,
            risk_measure_param=risk_measure_param,
            seed=seed,
        )

    def generate_scenario_data(
        self,
        δ_load_var=0.1,
        δ_pv_var=0.1,
        δ_b_var=0.1,
        number_of_scenarios=100,
        number_of_stages=3,
        seed_number=42,
    ):
        """Generate long-term scenarios with configurable parameters."""
        return generate_long_term_scenarios(
            nodes=self.grid_data_rm.node_data,
            δ_load_var=δ_load_var,
            δ_pv_var=δ_pv_var,
            δ_b_var=δ_b_var,
            number_of_scenarios=number_of_scenarios,
            number_of_stages=number_of_stages,
            seed_number=seed_number,
        )

    def create_expansion_request(
        self,
        planning_params=None,
        additional_params=None,
        scenario_data=None,
        bender_cuts=None,
    ):
        """Create expansion request with provided or default parameters."""
        if planning_params is None:
            planning_params = self.create_planning_params()
        if additional_params is None:
            additional_params = self.create_additional_params()
        if scenario_data is None:
            scenario_data = self.generate_scenario_data()
        if bender_cuts is None:
            bender_cuts = BenderCuts(cuts={})

        return dig_a_plan_to_expansion(
            grid_data=self.grid_data_rm,
            planning_params=planning_params,
            additional_params=additional_params,
            scenarios_data=scenario_data,
            bender_cuts=bender_cuts,
            scenarios_cache=self.cache_dir / "scenarios.json",
            bender_cuts_cache=self.cache_dir / "bender_cuts.json",
            optimization_config_cache=self.cache_dir / "optimization_config.json",
        )


class TestExpansionDataExporter(ExpansionTestBase):
    """Test class for expansion data exporter functionality."""

    def test_expansion_data_exporter(self):
        """Test the basic expansion data export functionality."""
        expansion_request = self.create_expansion_request()
        results = run_sddp(expansion_request, self.cache_dir)

        assert results is not None
        assert np.mean(results.objectives) == 0.0
        assert math.isclose(
            np.mean(
                [
                    results.simulations[ω][0].δ_load
                    for ω in range(len(results.simulations))
                ]
            ),
            0.049133186516891136,
            rel_tol=1e-6,
        )

    def test_expansion_with_different_stages(self):
        """Test expansion with different number of stages."""
        planning_params = self.create_planning_params(n_stages=5)
        scenario_data = self.generate_scenario_data(number_of_stages=5)
        expansion_request = self.create_expansion_request(
            planning_params=planning_params, scenario_data=scenario_data
        )
        results = run_sddp(expansion_request, self.cache_dir)

        assert results is not None
        assert len(results.simulations[0]) == 5

    def test_expansion_with_different_budget(self):
        """Test expansion with different initial budget."""
        planning_params = self.create_planning_params(initial_budget=50000)
        expansion_request = self.create_expansion_request(
            planning_params=planning_params
        )
        results = run_sddp(expansion_request, self.cache_dir)

        assert results is not None
        assert np.mean(results.objectives) == 0.0

    def test_expansion_with_cvar_risk_measure(self):
        """Test expansion with CVaR risk measure."""
        additional_params = self.create_additional_params(
            risk_measure_type=RiskMeasureType.CVAR, risk_measure_param=0.95
        )
        expansion_request = self.create_expansion_request(
            additional_params=additional_params
        )
        results = run_sddp(expansion_request, self.cache_dir)

        assert results is not None


class TestExpansionADMM(ExpansionTestBase):
    """Test class for ADMM-based expansion functionality."""

    def test_expansion_admm_input_setup(self):
        """Test basic ADMM configuration setup."""
        expansion_request = self.create_expansion_request()
        node_ids = [node.id for node in expansion_request.optimization.grid.nodes]
        edge_ids = [edge.id for edge in expansion_request.optimization.grid.edges]
        results = run_sddp(expansion_request, self.cache_dir)
        groups = self.simple_grid_groups
        admm = ADMM(groups=groups, grid_data=self.grid_data)
        for stage in range(expansion_request.optimization.planning_params.n_stages):
            for ω in range(len(expansion_request.scenarios.model_dump().keys())):
                admm.update_grid_data(
                    δ_load=results.simulations[ω][stage].δ_load,
                    δ_pv=results.simulations[ω][stage].δ_pv,
                    node_ids=node_ids,
                    δ_cap=results.simulations[ω][stage].δ_cap,
                    edge_ids=edge_ids,
                )
                admm.solve()
                break
            break
        assert True
