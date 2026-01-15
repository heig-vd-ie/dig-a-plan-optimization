from api.grid_cases import get_grid_case
from data_model.reconfiguration import CombinedInput, ReconfigurationOutput
from pipeline_reconfiguration import DigAPlanCombined


def run_combined(requests: CombinedInput) -> ReconfigurationOutput:
    _, base_grid_data = get_grid_case(
        grid=requests.grid,
        seed=requests.konfig.seed,
        profiles=requests.profiles,
        stu=requests.scenarios,
    )
    dig_a_plan = DigAPlanCombined(konfig=requests.konfig)
    dig_a_plan.add_grid_data(base_grid_data)
    dig_a_plan.solve_model(groups=requests.konfig.groups)  # one‐shot solve
    switches = dig_a_plan.result_manager.extract_switch_status()
    # Node voltages
    voltages = dig_a_plan.result_manager.extract_node_voltage()
    # Line currents
    currents = dig_a_plan.result_manager.extract_edge_current()
    taps = dig_a_plan.result_manager.extract_transformer_tap_position()
    result = ReconfigurationOutput(
        switches=switches.to_dicts(),
        voltages=voltages.to_dicts(),
        currents=currents.to_dicts(),
        taps=taps.to_dicts(),
    )

    return result
