import pandapower as pp
import polars as pl

from data_exporter.pandapower_to_dig_a_plan import pandapower_to_dig_a_plan_schema
from data_display.output_processing import compare_dig_a_plan_with_pandapower
from pipelines import DigAPlan
from pipelines.configs import CombinedConfig, PipelineType

from pipelines.model_managers.admm import PipelineModelManagerADMM
from pipelines.model_managers.bender import PipelineModelManagerBender
from pyomo_utility import extract_optimization_results


def test_combined_model_simple_example():

    net = pp.from_pickle("data/simple_grid.p")

    base_grid_data = pandapower_to_dig_a_plan_schema(net)

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
    assert edge_data.get_column("i_diff").abs().max() < 0.1  # type: ignore
    if isinstance(dig_a_plan.model_manager, PipelineModelManagerBender) or isinstance(
        dig_a_plan.model_manager, PipelineModelManagerADMM
    ):
        raise ValueError(
            "The model manager is not a Bender model manager, but a Combined model manager."
        )
    δ = extract_optimization_results(
        dig_a_plan.model_manager.combined_model_instance, "δ"
    )
    assert δ.filter(pl.col("δ") == 0).get_column("S").sort().to_list() == [
        21,
        24,
        26,
        33,
        34,
    ]
