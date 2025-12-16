from pydantic import BaseModel
from api.models import CombinedInput, ReconfigurationOutput
from api.grid_cases import get_grid_case
from experiments import *


def run_combined(input: CombinedInput) -> ReconfigurationOutput:
    net, base_grid_data = get_grid_case(
        grid=input.grid, seed=input.seed, stu=input.scenarios
    )
    config = CombinedConfig(
        verbose=True,
        big_m=1e3,
        ε=0.1,
        pipeline_type=PipelineType.COMBINED,
        γ_infeasibility=1.0,
        γ_admm_penalty=0.0,
    )
    dig_a_plan = DigAPlanCombined(config=config)
    dig_a_plan.add_grid_data(base_grid_data)
    dig_a_plan.solve_model(groups=input.groups)  # one‐shot solve
    switches = dig_a_plan.result_manager.extract_switch_status()
    # Node voltages
    voltages = dig_a_plan.result_manager.extract_node_voltage()
    # Line currents
    currents = dig_a_plan.result_manager.extract_edge_current()
    # Power flow
    powers = dig_a_plan.result_manager.extract_edge_active_power_flow()
    reactive_powers = dig_a_plan.result_manager.extract_edge_reactive_power_flow()
    taps = dig_a_plan.result_manager.extract_transformer_tap_position()
    if input.grid.pp_file == "examples/ieee-33/simple_grid.p":
        fig = plot_grid_from_pandapower(net=net, dap=dig_a_plan)
        node_data, edge_data = compare_dig_a_plan_with_pandapower(
            dig_a_plan=dig_a_plan, net=net
        )
    return ReconfigurationOutput(
        switches=switches.to_dicts(),
        voltages=voltages.to_dicts(),
        currents=currents.to_dicts(),
        taps=taps.to_dicts(),
    )
