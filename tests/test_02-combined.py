import pandapower as pp
import polars as pl

from local_data_exporter import pandapower_to_dig_a_plan_schema
from data_display.output_processing import compare_dig_a_plan_with_pandapower
from pipelines import DigAPlan
from pipelines.configs import CombinedConfig, PipelineType

from pipelines.model_managers.bender import PipelineModelManagerBender


def test_combined_model_simple_example():
    LOAD_FACTOR = 1
    TEST_CONFIG = [
        {"line_list": [], "switch_list": []},
        {"line_list": [6, 9], "switch_list": [25, 28]},
        {"line_list": [2, 6, 9], "switch_list": [21, 25, 28]},
        {"line_list": [16], "switch_list": [35]},
        {"line_list": [1], "switch_list": [20]},
        {"line_list": [10], "switch_list": [29]},
        {"line_list": [7, 11], "switch_list": [26, 30]},
    ]
    NB_TEST = 0

    net = pp.from_pickle("data/simple_grid.p")

    net["load"]["p_mw"] = net["load"]["p_mw"] * LOAD_FACTOR
    net["load"]["q_mvar"] = net["load"]["q_mvar"] * LOAD_FACTOR

    net["line"].loc[:, "max_i_ka"] = 1
    net["line"].loc[TEST_CONFIG[NB_TEST]["line_list"], "max_i_ka"] = 1e-2

    base_grid_data = pandapower_to_dig_a_plan_schema(net)

    config = CombinedConfig(
        verbose=False,
        big_m=1e2,
        factor_p=1e-3,
        factor_q=1e-3,
        factor_v=1,
        factor_i=1e-3,
        pipeline_type=PipelineType.COMBINED,
    )
    dig_a_plan = DigAPlan(config=config)

    dig_a_plan.add_grid_data(base_grid_data)
    dig_a_plan.solve_model()  # one‐shot solve

    # Switch status
    switches = dig_a_plan.result_manager.extract_switch_status()
    # Node voltages
    voltages = dig_a_plan.result_manager.extract_node_voltage()
    # Line currents
    currents = dig_a_plan.result_manager.extract_edge_current()

    node_data, edge_data = compare_dig_a_plan_with_pandapower(
        dig_a_plan=dig_a_plan, net=net
    )
    assert node_data.get_column("v_diff").abs().max() < 1e-6  # type: ignore
    assert edge_data.get_column("i_diff").abs().max() < 1e-3  # type: ignore
    if isinstance(dig_a_plan.model_manager, PipelineModelManagerBender):
        raise ValueError(
            "The model manager is not a Combined model manager, but a Bender model manager."
        )

    print(switches.filter(pl.col("open")).get_column("eq_fk").sort().to_list())
    assert switches.filter(pl.col("open")).get_column("eq_fk").sort().to_list() == [
        "switch 13",
        "switch 14",
        "switch 15",
        "switch 6",
        "switch 9",
    ]
