# %%
import os

os.chdir(os.getcwd().replace("/src", ""))
# %% import libraries
from examples import *

# %% set parameters

net = pp.from_pickle("data/simple_grid.p")
base_grid_data = pandapower_to_dig_a_plan_schema(net, taps=[99, 100, 101])

# %% initialize DigAPlan
config = BenderConfig(
    verbose=False,
    big_m=1e2,
    factor_p=1e-3,
    factor_q=1e-3,
    factor_v=1,
    factor_i=1e-3,
    master_relaxed=False,
    pipeline_type=PipelineType.BENDER,
)
dig_a_plan = DigAPlanBender(config=config)

# %% add grid data and solve models pipeline
dig_a_plan.add_grid_data(base_grid_data)
dig_a_plan.solve_model(max_iters=100)


# %% plot the results
fig = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.01,
    row_titles=["Slave objective", "Master objective"],
)
fig.add_trace(
    go.Scatter(
        go.Scatter(y=dig_a_plan.model_manager.slave_obj_list[1:]),
        mode="lines",
        name="Slave objective",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        go.Scatter(y=dig_a_plan.model_manager.master_obj_list[1:]),
        mode="lines",
        name="Master objective",
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(
        go.Scatter(y=dig_a_plan.model_manager.convergence_list[1:]),
        mode="lines",
        name="Convergence",
    ),
    row=3,
    col=1,
)
fig.update_layout(height=600, width=600, margin=dict(t=10, l=20, r=10, b=10))

# %% compare with pandapower
node_data, edge_data = compare_dig_a_plan_with_pandapower(
    dig_a_plan=dig_a_plan, net=net
)
plot_grid_from_pandapower(net=net, dig_a_plan=dig_a_plan)

# %% print(dig_a_plan.master_model_instance.objective.expr.to_string())
print(
    extract_optimization_results(dig_a_plan.model_manager.master_model_instance, "δ")
    .to_pandas()
    .to_string()
)

# %% extract and compare results
# Switch status
switches = dig_a_plan.result_manager.extract_switch_status()
# Node voltages
voltages = dig_a_plan.result_manager.extract_node_voltage()
# Line currents
currents = dig_a_plan.result_manager.extract_edge_current()
