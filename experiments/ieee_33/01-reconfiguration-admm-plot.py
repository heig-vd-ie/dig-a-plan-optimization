# %% Load Libraries
from experiments import *

# %% Load DAP States and Network
os.chdir(PROJECT_ROOT)
GRID_NAME = "test"
dap = load_dap_state(str(OUTPUT_ADMM_PATH / GRID_NAME))
dap_fixed = load_dap_state(str(OUTPUT_ADMM_PATH / f"{GRID_NAME}_fixed"))
net = joblib.load(str(OUTPUT_ADMM_PATH / f"{GRID_NAME}.joblib"))
results = load_obj_from_json(OUTPUT_ADMM_PATH / f"{GRID_NAME}_result.json")


# %% Inspect consensus and per-scenario deltas
LOG.info("\n=== ADMM consensus switch states (z) ===")
LOG.info(dap.model_manager.zδ_variable)
LOG.info(dap.model_manager.zζ_variable)

# %% Plot Distribution
nodal_variables = ["voltage", "p_curt_cons", "p_curt_prod"]
edge_variables = ["current", "p_flow", "q_flow"]
for variable in nodal_variables + edge_variables:
    DistributionVariable(
        daps={"ADMM": dap, "Normal Open": dap_fixed},  # type: ignore
        variable_name=variable,
        variable_type=("nodal" if variable in nodal_variables else "edge"),
    ).plot()

# %% Plot iteration of r_norm and s_norm
plot_admm_convergence(dap=dap)

# %%
plot_power_flow_results(dap=dap, node_size=5)
