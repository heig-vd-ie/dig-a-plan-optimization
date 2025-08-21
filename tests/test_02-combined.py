import pytest
import polars as pl
import math
from data_exporter.pandapower_to_dig_a_plan import pandapower_to_dig_a_plan_schema
from data_display.output_processing import compare_dig_a_plan_with_pandapower
from pipelines.reconfiguration import DigAPlanCombined

from pipelines.helpers.pyomo_utility import extract_optimization_results


class TestCombinedModel:

    @pytest.fixture(autouse=True)
    def setup_common_data(self, test_simple_grid, test_taps, test_combined_config):
        """Set up common test data and configurations."""
        self.net = test_simple_grid
        self.taps = test_taps
        self.combined_config = test_combined_config


class TestCombinedModelSimpleExample(TestCombinedModel):
    def test_combined_model_simple_example(self):

        base_grid_data = pandapower_to_dig_a_plan_schema(self.net, taps=self.taps)

        dig_a_plan = DigAPlanCombined(config=self.combined_config)

        dig_a_plan.add_grid_data(base_grid_data)
        dig_a_plan.solve_model()  # one‐shot solve

        # Switch status
        switches = dig_a_plan.result_manager.extract_switch_status()
        # Node voltages
        voltages = dig_a_plan.result_manager.extract_node_voltage()
        # Line currents
        currents = dig_a_plan.result_manager.extract_edge_current()
        taps = dig_a_plan.result_manager.extract_transformer_tap_position()

        node_data, edge_data = compare_dig_a_plan_with_pandapower(
            dig_a_plan=dig_a_plan, net=self.net
        )

        assert taps.get_column("tap_value").sort().to_list() == [100, 100]
        assert node_data.get_column("v_diff").abs().max() < 1e-1  # type: ignore
        assert math.isclose(edge_data.get_column("i_diff").abs().max(), 0.009211918701301954, rel_tol=1e-3, abs_tol=1e-4)  # type: ignore
        assert math.isclose(
            currents.get_column("i_pu").sum(),
            27.487386930375454,
            rel_tol=1e-3,
            abs_tol=1e-3,
        )
        assert math.isclose(
            voltages.get_column("v_pu").std(), 0.0031885507116687596, rel_tol=1e-3, abs_tol=1e-3  # type: ignore
        )
        δ = extract_optimization_results(
            dig_a_plan.model_manager.combined_model_instance, "δ"
        )
        assert len(δ.filter(pl.col("δ") == 0).get_column("S").sort().to_list()) == 5
