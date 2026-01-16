# %% Load Libraries
from experiments import *

# %% Load DAP State + Network + Results
os.chdir(PROJECT_ROOT)
GRID_NAME = "test"


dap = load_dap_state(str(OUTPUT_BENDER_PATH / GRID_NAME))
net = joblib.load(str(OUTPUT_BENDER_PATH / f"{GRID_NAME}.joblib"))
results = load_obj_from_json(OUTPUT_BENDER_PATH / f"{GRID_NAME}_result.json")

# %% Quick sanity checks
print("Loaded kind:", load_obj_from_json(OUTPUT_BENDER_PATH / GRID_NAME / "metadata.json").get("kind"))
print("Has slave_obj_list?", hasattr(dap.model_manager, "slave_obj_list"))
print("Has master_obj_list?", hasattr(dap.model_manager, "master_obj_list"))
print("Has convergence_list?", hasattr(dap.model_manager, "convergence_list"))

# %% Plot Bender convergence (slave/master/difference)
fig = make_subplots(rows=1, cols=1)

slave = getattr(dap.model_manager, "slave_obj_list", [])
master = getattr(dap.model_manager, "master_obj_list", [])
diff = getattr(dap.model_manager, "convergence_list", [])

# Keep your original behavior of skipping the first element if you want
if len(slave) > 1:
    fig.add_trace(go.Scatter(y=slave[1:], mode="lines", name="Slave objective"), row=1, col=1)
elif len(slave) == 1:
    fig.add_trace(go.Scatter(y=slave, mode="lines", name="Slave objective"), row=1, col=1)

if len(master) > 1:
    fig.add_trace(go.Scatter(y=master[1:], mode="lines", name="Master objective"), row=1, col=1)
elif len(master) == 1:
    fig.add_trace(go.Scatter(y=master, mode="lines", name="Master objective"), row=1, col=1)

if len(diff) > 1:
    fig.add_trace(
        go.Scatter(y=diff[1:], mode="lines", line=dict(dash="dot"), name="Difference"),
        row=1,
        col=1,
    )
elif len(diff) == 1:
    fig.add_trace(
        go.Scatter(y=diff, mode="lines", line=dict(dash="dot"), name="Difference"),
        row=1,
        col=1,
    )

fig.update_layout(
    height=600,
    width=1200,
    margin=dict(t=10, l=20, r=10, b=10),
    legend=dict(
        x=0.70,
        y=0.98,
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="rgba(0,0,0,0.2)",
        borderwidth=1,
    ),
    xaxis_title="Iteration",
    yaxis_title="Objective Value",
)

os.makedirs(".cache/figs", exist_ok=True)
fig.write_html(".cache/figs/bender-convergence.html")
fig.write_image(".cache/figs/bender-convergence.svg", format="svg")

# %% Compare with pandapower + plot grid
node_data, edge_data = compare_dig_a_plan_with_pandapower(dig_a_plan=dap, net=net) #type: ignore

plot_grid_from_pandapower(dap=dap)
plot_grid_from_pandapower(dap=dap, color_by_results=True)

# %% Plot power flow results 
plot_power_flow_results(dap=dap, node_size=5)

# %% Print switch status (from saved parquet via MockResultManager)
print("\n=== Switch status ===")
print(dap.result_manager.extract_switch_status().to_pandas().to_string())
