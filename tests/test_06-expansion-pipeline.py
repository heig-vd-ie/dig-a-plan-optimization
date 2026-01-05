import datetime
import math
import numpy as np
import pytest
from data_exporter.pp_to_dap import (
    pandapower_to_dig_a_plan_schema_with_scenarios,
)
from pipeline_expansion.algorithm import ExpansionAlgorithm
from data_model.sddp import Node, RiskMeasureType


class ExpansionTestBase:
    """Base class for expansion pipeline tests with common setup."""

    @pytest.fixture(autouse=True)
    def setup_common_data(
        self, test_simple_grid, test_cache_dir, test_simple_grid_groups
    ):
        """Set up common test data and configurations."""
        self.net = test_simple_grid
        self.grid_data = pandapower_to_dig_a_plan_schema_with_scenarios(self.net)
        self.cache_dir = test_cache_dir
        self.simple_grid_groups = test_simple_grid_groups
        self.expansion_algorithm = ExpansionAlgorithm(
            grid_data=self.grid_data,
            each_task_memory=1024,
            time_now=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
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
        δ_load_var: float = 5.0
        δ_pv_var: float = 1.0
        nodes = [
            Node(id=node["node_id"])
            for node in self.grid_data.node_data.iter_rows(named=True)
        ]
        self.expansion_algorithm.create_expansion_request()
        self.expansion_algorithm.create_planning_params(n_stages=5)
        self.expansion_algorithm.create_scenario_data(
            nodes=self.expansion_algorithm.expansion_request.optimization.grid.nodes,
            number_of_stages=5,
            load_potential={node.id: δ_load_var for node in nodes},
            pv_potential={node.id: δ_pv_var for node in nodes},
        )
        self.expansion_algorithm.create_out_of_sample_scenario_data(
            nodes=self.expansion_algorithm.expansion_request.optimization.grid.nodes,
            number_of_stages=5,
            load_potential={node.id: δ_load_var for node in nodes},
            pv_potential={node.id: δ_pv_var for node in nodes},
        )
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
        assert np.abs(np.mean(results.objectives) - 8817.65442176871) <= 1e-3

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
