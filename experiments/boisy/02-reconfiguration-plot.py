# %% Load Libraries
from experiments import *

# %% Load DAP States and Network
os.chdir(PROJECT_ROOT)
dap = load_dap_state(str(OUTPUT_ADMM_PATH / "test"))
dap_fixed = load_dap_state(str(OUTPUT_ADMM_PATH / "test_fixed"))
net = joblib.load(str(OUTPUT_ADMM_PATH / "test.joblib"))
results = load_obj_from_json(OUTPUT_ADMM_PATH / "test_result.json")


# %% Inspect consensus and per-scenario deltas
print("\n=== ADMM consensus switch states (z) ===")
print(dap.model_manager.zδ_variable)
print(dap.model_manager.zζ_variable)

# %% Plot Distribution
nodal_variables = [
    "voltage",
    "p_curt_cons",
    "p_curt_prod",
    "q_curt_cons",
    "q_curt_prod",
]
edge_variables = [
    "current",
    "p_flow",
    "q_flow",
]
for variable in nodal_variables + edge_variables:
    DistributionVariable(
        daps={"ADMM": dap, "Normal Open": dap_fixed},  # type: ignore
        variable_name=variable,
        variable_type=("nodal" if variable in nodal_variables else "edge"),
    ).plot()

# %% Plot iteration of r_norm and s_norm
plot_admm_convergence(dap=dap)

# %%
plot_grid_from_pandapower(dap=dap, from_z=True, color_by_results=True, node_size=6)  # type: ignore

# %% Plot fixed switches
plot_grid_from_pandapower(dap=dap_fixed, from_z=True, color_by_results=True, node_size=6)  # type: ignore

# %%
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
