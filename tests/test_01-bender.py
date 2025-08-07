import pandapower as pp
import polars as pl

from data_exporter.pandapower_to_dig_a_plan import pandapower_to_dig_a_plan_schema
from data_display.output_processing import compare_dig_a_plan_with_pandapower
from pipelines import DigAPlanBender
from pipelines.configs import BenderConfig, PipelineType

from pyomo_utility import extract_optimization_results


def test_bender_model_simple_example():

    net = pp.from_pickle("data/simple_grid.p")
    base_grid_data = pandapower_to_dig_a_plan_schema(
        net, taps=[95, 98, 99, 100, 101, 102, 105]
    )

    config = BenderConfig(
        verbose=False,
        big_m=1e2,
        factor_p=1e-3,
        factor_q=1e-3,
        factor_v=1,
        factor_i=1e-3,
        master_relaxed=False,
        pipeline_type=PipelineType.BENDER,
    )
    dig_a_plan = DigAPlanBender(config=config)

    dig_a_plan.add_grid_data(base_grid_data)
    dig_a_plan.solve_model(max_iters=100)
    node_data, edge_data = compare_dig_a_plan_with_pandapower(
        dig_a_plan=dig_a_plan, net=net
    )
    assert node_data.get_column("v_diff").abs().max() < 1e-6  # type: ignore
    assert edge_data.get_column("i_diff").abs().max() < 5e-3  # type: ignore

    δ = extract_optimization_results(
        dig_a_plan.model_manager.master_model_instance, "δ"
    )
    assert δ.filter(pl.col("δ") == 0).get_column("S").sort().to_list() == [
        23,
        28,
        32,
        33,
        34,
    ]
