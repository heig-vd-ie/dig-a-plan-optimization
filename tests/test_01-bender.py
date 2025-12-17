import polars as pl
import pytest

from data_exporter.kace_to_dap import kace4reconfiguration
from data_display.output_processing import compare_dig_a_plan_with_pandapower
from data_model.reconfiguration import BenderInput
from pipelines.reconfiguration import DigAPlanBender

from pipelines.helpers.pyomo_utility import extract_optimization_results


class BenderTestCase:
    """Base class for Bender test cases."""

    @pytest.fixture(autouse=True)
    def setup_common_data(
        self, test_simple_grid, test_bender_config, bender_input_payload: BenderInput
    ):
        """Set up common test data and configurations."""
        self.net = test_simple_grid
        self.bender_config = test_bender_config
        self.bender_input_payload = bender_input_payload


class TestBenderModel(BenderTestCase):
    def test_bender_model_simple_example(self):

        base_grid_data = kace4reconfiguration(
            self.bender_input_payload.grid,
            self.bender_input_payload.load_profiles,
            self.bender_input_payload.scenarios,
            self.bender_input_payload.seed,
        )

        dig_a_plan = DigAPlanBender(konfig=self.bender_config)

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
