from api import *


class CombinedInput(GridCaseModel):
    grid_case: GridCase
    max_iters: int = 100


class CombinedOutput(ReconfigurationOutput):
    pass


def run_combined(input: CombinedInput) -> CombinedOutput:
    net = get_grid_case(input.grid_case)
    base_grid_data = pandapower_to_dig_a_plan_schema(
        net=net,
        taps=input.taps,
        v_bounds=input.v_bounds,
        p_bounds=input.p_bounds,
        q_bounds=input.q_bounds,
        number_of_random_scenarios=input.number_of_random_scenarios,
        v_min=input.v_min,
        v_max=input.v_max,
        seed=input.seed,
    )
    config = CombinedConfig(
        verbose=True,
        big_m=1e3,
        ε=1,
        pipeline_type=PipelineType.COMBINED,
        γ_infeasibility=1.0,
        γ_admm_penalty=0.0,
    )
    dig_a_plan = DigAPlanCombined(config=config)
    dig_a_plan.add_grid_data(base_grid_data)
    dig_a_plan.solve_model()
    switches = dig_a_plan.result_manager.extract_switch_status()
    # Node voltages
    voltages = dig_a_plan.result_manager.extract_node_voltage()
    # Line currents
    currents = dig_a_plan.result_manager.extract_edge_current()
    # Power flow
    powers = dig_a_plan.result_manager.extract_edge_active_power_flow()
    reactive_powers = dig_a_plan.result_manager.extract_edge_reactive_power_flow()
    taps = dig_a_plan.result_manager.extract_transformer_tap_position()
    print(taps)
    fig = plot_grid_from_pandapower(net=net, dap=dig_a_plan)
    node_data, edge_data = compare_dig_a_plan_with_pandapower(
        dig_a_plan=dig_a_plan, net=net
    )
    return CombinedOutput(
        switches=switches.to_dicts(),
        voltages=voltages.to_dicts(),
        currents=currents.to_dicts(),
        taps=taps.to_dicts(),
    )
