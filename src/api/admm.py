import copy
import joblib
from pathlib import Path
from data_model.reconfiguration import ADMMInput, ReconfigurationOutput
from api.grid_cases import get_grid_case
from pipeline_reconfiguration import DigAPlanADMM
from data_exporter.mock_dap import save_dap_state
from konfig import settings


def run_admm(requests: ADMMInput) -> ReconfigurationOutput:
    net, base_grid_data = get_grid_case(
        grid=requests.grid, seed=requests.konfig.seed, stu=requests.scenarios
    )
    dap = DigAPlanADMM(konfig=requests.konfig)
    dap.add_grid_data(base_grid_data)
    dap.solve_model(groups=requests.konfig.groups)
    # Fixed switches solution (for distribution plots)
    dap_fixed = copy.deepcopy(dap)
    dap_fixed.solve_model(fixed_switches=True)

    switches = dap.model_manager.zδ_variable
    taps = dap.model_manager.zζ_variable
    voltages = dap.result_manager.extract_node_voltage(scenario=0)
    currents = dap.result_manager.extract_edge_current(scenario=0)

    save_dap_state(dap, str(Path(settings.cache.figures) / requests.grid.name))
    save_dap_state(
        dap_fixed, str(Path(settings.cache.figures) / (requests.grid.name + "_fixed"))
    )
    joblib.dump(
        net, str(Path(settings.cache.figures) / (requests.grid.name + ".joblib"))
    )

    return ReconfigurationOutput(
        switches=switches.to_dicts(),
        voltages=voltages.to_dicts(),
        currents=currents.to_dicts(),
        taps=taps.to_dicts(),
    )
