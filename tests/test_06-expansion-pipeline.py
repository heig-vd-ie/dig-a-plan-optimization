import math
import numpy as np
import pytest
from api.grid_cases import get_grid_case
from data_model.expansion import SDDPConfig
from pipeline_expansion.algorithm import ExpansionAlgorithm
from data_model.sddp import RiskMeasureType
from pipeline_reconfiguration import ADMMConfig


class ExpansionTestBase:
    """Base class for expansion pipeline tests with common setup."""

    @pytest.fixture(autouse=True)
    def setup_common_data(
        self,
        test_simple_grid,
        test_cache_dir,
        test_simple_grid_groups,
        test_seed,
        test_short_term_uncertainty_random,
    ):
        """Set up common test data and configurations."""
        self.grid = test_simple_grid
        self.seed = test_seed
        self.stu = test_short_term_uncertainty_random
        _, self.grid_data = get_grid_case(grid=self.grid, seed=self.seed, stu=self.stu)
        self.cache_dir = test_cache_dir
        self.simple_grid_groups = test_simple_grid_groups
        self.load_potential = {
            node: 5.0 for node in self.grid_data.node_data["node_id"].to_list()
        }
        self.pv_potential = {
            node: 1.0 for node in self.grid_data.node_data["node_id"].to_list()
        }
        self.each_task_memory = 1e8

        self.expansion_algorithm = ExpansionAlgorithm(
            grid_data=self.grid_data,
            load_potential=self.load_potential,
            pv_potential=self.pv_potential,
            each_task_memory=self.each_task_memory,
            admm_config=ADMMConfig(),
            sddp_config=SDDPConfig(
                n_stages=3,
                n_scenarios=100,
                n_simulations=100,
                n_optimizations=10,
                initial_budget=1e6,
                discount_rate=0.05,
                years_per_stage=1,
                iterations=10,
                expansion_line_cost_per_km_kw=1e3,
                expansion_transformer_cost_per_kw=1e3,
                penalty_cost_per_consumption_kw=1e3,
                penalty_cost_per_production_kw=1e3,
                δ_b_var=1000,
            ),
            time_now="run_test",
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
        assert np.abs(np.mean(results.objectives) - 8817.65442176871) <= 1e-3
        assert math.isclose(
            np.mean(
                [
                    results.simulations[ω][0].δ_load
                    for ω in range(len(results.simulations))
                ]
            ),
            0.3333333333333333,
            rel_tol=1e-1,
        )

    def test_expansion_with_different_stages(self):
        """Test expansion with different number of stages."""
        self.expansion_algorithm.create_expansion_request()
        self.expansion_algorithm.create_planning_params(n_stages=5)
        self.expansion_algorithm.scenario_data = self.expansion_algorithm.create_scenario_data(
            nodes=self.expansion_algorithm.expansion_request.optimization.grid.nodes,
            load_potential=self.load_potential,
            pv_potential=self.pv_potential,
            n_stages=5,
        )
        self.expansion_algorithm.out_of_sample_scenarios = self.expansion_algorithm.create_scenario_data(
            nodes=self.expansion_algorithm.expansion_request.optimization.grid.nodes,
            load_potential=self.load_potential,
            pv_potential=self.pv_potential,
            n_stages=5,
        )
        self.expansion_algorithm.create_expansion_request()
        results = self.expansion_algorithm.run_sddp()

        assert results is not None
        assert len(results.simulations[0]) == 5

    def test_expansion_with_different_budget(self):
        """Test expansion with different initial budget."""
        self.expansion_algorithm.sddp_config.initial_budget = 50000
        self.expansion_algorithm.create_planning_params()
        self.expansion_algorithm.create_expansion_request()
        results = self.expansion_algorithm.run_sddp()

        assert results is not None
        assert np.abs(np.mean(results.objectives) - 8817.65442176871) <= 1e-3

    def test_expansion_with_cvar_risk_measure(self):
        """Test expansion with CVaR risk measure."""
        self.expansion_algorithm.create_additional_params(
            sddp_config=SDDPConfig(
                risk_measure_type=RiskMeasureType.CVAR, risk_measure_param=0.95
            )
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
