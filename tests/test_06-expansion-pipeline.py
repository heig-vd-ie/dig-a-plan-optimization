import math
import numpy as np
import pytest
from data_exporter.pandapower_to_dig_a_plan import pandapower_to_dig_a_plan_schema
from pipelines.expansion.algorithm import ExpansionAlgorithm
from pipelines.expansion.models.request import RiskMeasureType


class ExpansionTestBase:
    """Base class for expansion pipeline tests with common setup."""

    @pytest.fixture(autouse=True)
    def setup_common_data(
        self, test_simple_grid, test_cache_dir, test_simple_grid_groups
    ):
        """Set up common test data and configurations."""
        self.net = test_simple_grid
        self.grid_data = pandapower_to_dig_a_plan_schema(self.net)
        self.cache_dir = test_cache_dir
        self.simple_grid_groups = test_simple_grid_groups
        self.expansion_algorithm = ExpansionAlgorithm(
            grid_data=self.grid_data,
            each_task_memory=1024,
            cache_dir=self.cache_dir,
            just_test=True,
        )


class TestExpansionDataExporter(ExpansionTestBase):
    """Test class for expansion data exporter functionality."""

    def test_expansion_data_exporter(self):
        """Test the basic expansion data export functionality."""
        self.expansion_algorithm.create_expansion_request()
        results = self.expansion_algorithm.run_sddp()

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
            rel_tol=1e-1,
        )

    def test_expansion_with_different_stages(self):
        """Test expansion with different number of stages."""
        self.expansion_algorithm.create_planning_params(n_stages=5)
        self.expansion_algorithm.create_scenario_data(number_of_stages=5)
        self.expansion_algorithm.create_expansion_request()
        results = self.expansion_algorithm.run_sddp()

        assert results is not None
        assert len(results.simulations[0]) == 5

    def test_expansion_with_different_budget(self):
        """Test expansion with different initial budget."""
        self.expansion_algorithm.create_planning_params(initial_budget=50000)
        self.expansion_algorithm.create_expansion_request()
        results = self.expansion_algorithm.run_sddp()

        assert results is not None
        assert np.mean(results.objectives) == 0.0

    def test_expansion_with_cvar_risk_measure(self):
        """Test expansion with CVaR risk measure."""
        self.expansion_algorithm.create_additional_params(
            risk_measure_type=RiskMeasureType.CVAR, risk_measure_param=0.95
        )
        self.expansion_algorithm.create_expansion_request()
        results = self.expansion_algorithm.run_sddp()

        assert results is not None


class TestExpansionADMM(ExpansionTestBase):
    """Test class for ADMM-based expansion functionality."""

    def test_expansion_admm_input_setup(self):
        """Test basic ADMM configuration setup."""
        self.expansion_algorithm.run_pipeline()
        assert True
