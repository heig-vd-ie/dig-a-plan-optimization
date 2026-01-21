import joblib
from pathlib import Path

from api.grid_cases import get_grid_case
from data_model.reconfiguration import CombinedInput, ReconfigurationOutput
from helpers.json import save_obj_to_json
from pipeline_reconfiguration import DigAPlanCombined
from data_exporter.mock_dap import save_dap_state
from konfig import settings


def run_combined(requests: CombinedInput) -> ReconfigurationOutput:
    net, base_grid_data = get_grid_case(
        grid=requests.grid,
        seed=requests.konfig.seed,
        profiles=requests.profiles,
        stu=requests.scenarios,
    )
    dap = DigAPlanCombined(konfig=requests.konfig)
    dap.add_grid_data(base_grid_data)
    dap.solve_model(groups=requests.konfig.groups)  # one‐shot solve
    # switch statuses
    switches = dap.result_manager.extract_switch_status()
    # Node voltages
    voltages = dap.result_manager.extract_node_voltage()
    # Line currents
    currents = dap.result_manager.extract_edge_current()
    taps = dap.result_manager.extract_transformer_tap_position()
    result = ReconfigurationOutput(
        switches=switches.to_dicts(),
        voltages=voltages.to_dicts(),
        currents=currents.to_dicts(),
        taps=taps.to_dicts(),
    )

    if requests.to_save:
        save_dap_state(
            dap,
            str(Path(settings.cache.outputs_combined) / requests.grid.name),
        )
        joblib.dump(
            net,
            str(Path(settings.cache.outputs_combined) / (requests.grid.name + ".joblib")),
        )
        save_obj_to_json(
            result,
            Path(settings.cache.outputs_combined)
            / (requests.grid.name + "_result.json"),
        )  
    return result
