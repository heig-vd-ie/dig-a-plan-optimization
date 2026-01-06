from data_model.reconfiguration import ADMMInput, ReconfigurationOutput
from api.grid_cases import get_grid_case
from experiments import *


def run_admm(requets: ADMMInput) -> ReconfigurationOutput:
    net, base_grid_data = get_grid_case(
        grid=requets.grid, seed=requets.konfig.seed, stu=requets.scenarios
    )
    dap = DigAPlanADMM(konfig=requets.konfig)
    dap.add_grid_data(base_grid_data)
    dap.solve_model(groups=requets.konfig.groups)
    # Fixed switches solution (for distribution plots)
    dap_fixed = copy.deepcopy(dap)
    dap_fixed.solve_model(fixed_switches=True)

    switches = dap.model_manager.zδ_variable
    taps = dap.model_manager.zζ_variable
    voltages = dap.result_manager.extract_node_voltage(scenario=0)
    currents = dap.result_manager.extract_edge_current(scenario=0)

    save_dap_state(dap, ".cache/figs/boisy_dap")
    save_dap_state(dap_fixed, ".cache/figs/boisy_dap_fixed")
    joblib.dump(net, ".cache/figs/boisy_net.joblib")

    return ReconfigurationOutput(
        switches=switches.to_dicts(),
        voltages=voltages.to_dicts(),
        currents=currents.to_dicts(),
        taps=taps.to_dicts(),
    )
