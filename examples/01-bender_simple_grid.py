# %% import libraries
from logging import config
import os
import pandapower as pp
import plotly.graph_objs as go

from data_connector import pandapower_to_dig_a_plan_schema
from data_display.grid_plotting import plot_grid_from_pandapower
from data_display.output_processing import compare_dig_a_plan_with_pandapower
from pipelines import DigAPlan
from pipelines.configs import BenderConfig, PipelineType

from pipelines.model_managers.combined import PipelineModelManagerCombined
from pyomo_utility import extract_optimization_results
from plotly.subplots import make_subplots

os.chdir(os.getcwd().replace("/src", ""))
os.environ["GRB_LICENSE_FILE"] = os.environ["HOME"] + "/gurobi_license/gurobi.lic"

# %% set parameters
LOAD_FACTOR = 1
TEST_CONFIG = [
    {"line_list": [], "switch_list": []},
    {"line_list": [6, 9], "switch_list": [25, 28]},
    {"line_list": [2, 6, 9], "switch_list": [21, 25, 28]},
    {"line_list": [16], "switch_list": [35]},
    {"line_list": [1], "switch_list": [20]},
    {"line_list": [10], "switch_list": [29]},
    {"line_list": [7, 11], "switch_list": [26, 30]},
]
NB_TEST = 0

net = pp.from_pickle("data/simple_grid.p")

net["load"]["p_mw"] = net["load"]["p_mw"] * LOAD_FACTOR
net["load"]["q_mvar"] = net["load"]["q_mvar"] * LOAD_FACTOR

net["line"].loc[:, "max_i_ka"] = 1
net["line"].loc[TEST_CONFIG[NB_TEST]["line_list"], "max_i_ka"] = 1e-2
# %% transform the pandapower grid to DigAPlan schema
base_grid_data = pandapower_to_dig_a_plan_schema(net)

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
dig_a_plan = DigAPlan(config=config)

# %% add grid data and solve models pipeline
dig_a_plan.add_grid_data(base_grid_data)
dig_a_plan.solve_model(max_iters=100)
if isinstance(dig_a_plan.model_manager, PipelineModelManagerCombined):
    raise ValueError(
        "The model manager is not a Bender model manager, but a Combined model manager."
    )


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
    extract_optimization_results(
        dig_a_plan.model_manager.master_model_instance, "delta"
    )
    .to_pandas()
    .to_string()
)
