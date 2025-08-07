import pandapower as pp
import polars as pl
import math
from data_exporter.pandapower_to_dig_a_plan import pandapower_to_dig_a_plan_schema
from data_display.output_processing import compare_dig_a_plan_with_pandapower
from pipelines import DigAPlanCombined
from pipelines.configs import CombinedConfig, PipelineType

from pyomo_utility import extract_optimization_results


def test_combined_model_simple_example():

    net = pp.from_pickle("data/simple_grid.p")

    base_grid_data = pandapower_to_dig_a_plan_schema(
        net, taps=[95, 98, 99, 100, 101, 102, 105]
    )

    config = CombinedConfig(
        verbose=False,
        big_m=1e3,
        γ_infeasibility=1.0,
        factor_p=1e-3,
        factor_q=1e-3,
        factor_v=1,
        factor_i=1e-3,
        pipeline_type=PipelineType.COMBINED,
    )
    dig_a_plan = DigAPlanCombined(config=config)

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
        dig_a_plan=dig_a_plan, net=net
    )

    assert taps.get_column("tap_value").sort().to_list() == [100, 100]
    assert node_data.get_column("v_diff").abs().max() < 1e-6  # type: ignore
    assert math.isclose(edge_data.get_column("i_diff").abs().max(), 0.001668, rel_tol=1e-3, abs_tol=1e-4)  # type: ignore
    assert math.isclose(
        currents.get_column("i_pu").sum(), 13.6158519, rel_tol=1e-3, abs_tol=1e-3
    )
    assert math.isclose(
        voltages.get_column("v_pu").std(), 0.0061036143174779955, rel_tol=1e-3, abs_tol=1e-3  # type: ignore
    )
    δ = extract_optimization_results(
        dig_a_plan.model_manager.combined_model_instance, "δ"
    )
    assert δ.filter(pl.col("δ") == 0).get_column("S").sort().to_list() == [
        24,
        25,
        32,
        33,
        34,
    ]
