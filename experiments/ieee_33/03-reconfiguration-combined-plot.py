# %% 

from experiments import *

os.chdir(PROJECT_ROOT)
GRID_NAME = "test"

dap = load_dap_state(str(OUTPUT_COMBINED_PATH / GRID_NAME))
net = joblib.load(str(OUTPUT_COMBINED_PATH / f"{GRID_NAME}.joblib"))
results = load_obj_from_json(OUTPUT_COMBINED_PATH / f"{GRID_NAME}_result.json")


# %% Compare with pandapower + plot grid
node_data, edge_data = compare_dig_a_plan_with_pandapower(dig_a_plan=dap, net=net) #type: ignore

plot_grid_from_pandapower(dap=dap)
plot_grid_from_pandapower(dap=dap, color_by_results=True)

# %% Plot power flow results
plot_power_flow_results(dap=dap, node_size=5)

# %% Print switch status
print("\n=== Switch status ===")
print(dap.result_manager.extract_switch_status().to_pandas().to_string())

