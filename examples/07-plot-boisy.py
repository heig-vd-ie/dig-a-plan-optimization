# %%
import os

os.chdir(os.getcwd().replace("/src", ""))

# %%
from examples import *

# %%
dap = load_dap_state(".cache/output/boisy_dap")
dap_fixed = load_dap_state(".cache/output/boisy_dap_fixed")

net = joblib.load(".cache/output/boisy_net.joblib")

base_grid_data = NodeEdgeModel(
    node_data=dap.data_manager.node_data,  # type: ignore
    edge_data=dap.data_manager.edge_data,  # type: ignore
    load_data={},
)

# %%
switches = dap.result_manager.extract_switch_status()  # type: ignore
# Node voltages
voltages = dap.result_manager.extract_node_voltage()  # type: ignore
# Line currents
currents = dap.result_manager.extract_edge_current()  # type: ignore


plot_power_flow_results(
    base_grid_data=base_grid_data,
    switches=switches,
    voltages=voltages,
    currents=currents,
    node_size=5,
)

# %% Plot fixed switches
switches = dap_fixed.result_manager.extract_switch_status()  # type: ignore
# Node voltages
voltages = dap_fixed.result_manager.extract_node_voltage()  # type: ignore
# Line currents
currents = dap_fixed.result_manager.extract_edge_current()  # type: ignore


plot_power_flow_results(
    base_grid_data=base_grid_data,
    switches=switches,
    voltages=voltages,
    currents=currents,
    node_size=5,
)
