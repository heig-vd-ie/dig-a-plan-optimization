import joblib
from pathlib import Path

from api.grid_cases import get_grid_case
from data_model.reconfiguration import BenderInput, ReconfigurationOutput
from helpers.json import save_obj_to_json
from pipeline_reconfiguration import DigAPlanBender
from data_exporter.mock_dap import save_dap_state
from konfig import settings


def run_bender(requests: BenderInput) -> ReconfigurationOutput:
    net, base_grid_data = get_grid_case(
        grid=requests.grid,
        seed=requests.konfig.seed,
        stu=requests.scenarios,
        profiles=requests.profiles,
    )
    dap = DigAPlanBender(konfig=requests.konfig)
    dap.add_grid_data(base_grid_data)
    dap.solve_model(max_iters=requests.konfig.max_iters)

    switches = dap.result_manager.extract_switch_status()
    voltages = dap.result_manager.extract_node_voltage()
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
            str(Path(settings.cache.outputs_bender) / requests.grid.name),
        )
        joblib.dump(
            net,
            str(Path(settings.cache.outputs_bender) / (requests.grid.name + ".joblib")),
        )
        save_obj_to_json(
            result,
            Path(settings.cache.outputs_bender)
            / (requests.grid.name + "_result.json"),
        )  
    return result     
