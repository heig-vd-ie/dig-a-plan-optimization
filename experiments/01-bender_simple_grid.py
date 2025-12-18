# %%
import os

os.chdir(os.getcwd().replace("/src", ""))
from api.bender import run_bender
from experiments import *

# %% set parameters
kace1 = BenderInput(
    **load_obj_from_json(Path("examples/payloads/reconfiguration/ex1-bender.json"))
)
net = pp.from_pickle(kace1.grid.pp_file)
bender_output, dap = run_bender(kace1)

# %% plot the results
fig = make_subplots(rows=1, cols=1)
fig.add_trace(
    go.Scatter(
        go.Scatter(y=dap.model_manager.slave_obj_list[1:]),
        mode="lines",
        name="Slave objective",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        go.Scatter(y=dap.model_manager.master_obj_list[1:]),
        mode="lines",
        name="Master objective",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        go.Scatter(y=dap.model_manager.convergence_list[1:]),
        mode="lines",
        line=dict(dash="dot"),
        name="Difference",
    ),
    row=1,
    col=1,
)
fig.update_layout(
    height=600,
    width=1200,
    margin=dict(t=10, l=20, r=10, b=10),
    legend=dict(
        x=0.70,  # Position legend inside the plot area
        y=0.98,  # Position at top-left
        bgcolor="rgba(255,255,255,0.8)",  # Semi-transparent white background
        bordercolor="rgba(0,0,0,0.2)",  # Light border
        borderwidth=1,
    ),
    xaxis_title="Iteration",
    yaxis_title="Objective Value",
)
os.makedirs(".cache/figs", exist_ok=True)
fig.write_html(".cache/figs/bender-convergence.html")
fig.write_image(".cache/figs/bender-convergence.svg", format="svg")

# %% compare with pandapower
node_data, edge_data = compare_dig_a_plan_with_pandapower(dig_a_plan=dap, net=net)
# %%
plot_grid_from_pandapower(dap=dap)

# %%
plot_grid_from_pandapower(dap=dap, color_by_results=True)


# %% print(dig_a_plan.master_model_instance.objective.expr.to_string())
print(
    extract_optimization_results(dap.model_manager.master_model_instance, "δ")
    .to_pandas()
    .to_string()
)
