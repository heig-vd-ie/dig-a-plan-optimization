import polars as pl
import pytest

from data_exporter.pp_to_dap import (
    pandapower_to_dig_a_plan_schema_with_scenarios,
)
from data_display.output_processing import compare_dig_a_plan_with_pandapower
from pipeline_reconfiguration import DigAPlanBender

from helpers.pyomo import extract_optimization_results


class BenderTestCase:
    """Base class for Bender test cases."""

    @pytest.fixture(autouse=True)
    def setup_common_data(self, test_simple_grid, test_bender_config):
        """Set up common test data and configurations."""
        self.net = test_simple_grid
        self.bender_config = test_bender_config


class TestBenderModel(BenderTestCase):
    def test_bender_model_simple_example(self):

        base_grid_data = pandapower_to_dig_a_plan_schema_with_scenarios(self.net)

        dig_a_plan = DigAPlanBender(config=self.bender_config)

        dig_a_plan.add_grid_data(base_grid_data)
        dig_a_plan.solve_model(max_iters=100)
        node_data, edge_data = compare_dig_a_plan_with_pandapower(
            dig_a_plan=dig_a_plan, net=self.net
        )
        assert node_data.get_column("v_diff").abs().max() < 1e-1  # type: ignore
        assert edge_data.get_column("i_diff").abs().max() < 1e-1  # type: ignore

        δ = extract_optimization_results(
            dig_a_plan.model_manager.master_model_instance, "δ"
        )
        assert len(δ.filter(pl.col("δ") == 0).get_column("S").sort().to_list()) == 5
