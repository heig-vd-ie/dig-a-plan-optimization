from data_model.reconfiguration import BenderInput, ReconfigurationOutput
from experiments import *
from api.grid_cases import get_grid_case


def run_bender(requests: BenderInput) -> ReconfigurationOutput:
    net, base_grid_data = get_grid_case(
        grid=requests.grid, seed=requests.konfig.seed, stu=requests.scenarios
    )
    dig_a_plan = DigAPlanBender(konfig=requests.konfig)
    dig_a_plan.add_grid_data(base_grid_data)
    dig_a_plan.solve_model(max_iters=requests.konfig.max_iters)

    switches = dig_a_plan.result_manager.extract_switch_status()
    voltages = dig_a_plan.result_manager.extract_node_voltage()
    currents = dig_a_plan.result_manager.extract_edge_current()
    taps = dig_a_plan.result_manager.extract_transformer_tap_position()
    return ReconfigurationOutput(
        switches=switches.to_dicts(),
        voltages=voltages.to_dicts(),
        currents=currents.to_dicts(),
        taps=taps.to_dicts(),
    )
