

# %%
import os
import pandapower as pp
import pandas as pd
from plotly.graph_objects import Figure
import plotly.graph_objs as go
import polars as pl
from polars import col as c

from pipelines.dig_a_plan import DigAPlan
from polars import selectors as cs 
# from pipelines.dig_a_plan_d_model import DigAPlan
from data_display.grid_plotting import plot_grid_from_pandapower
from data_display.output_processing import compare_dig_a_plan_with_pandapower
from data_connector import pandapower_to_dig_a_plan_schema
from general_function import pl_to_dict

from pyomo_utility import extract_optimization_results
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

os.chdir(os.getcwd().replace("/src", ""))
os.environ['GRB_LICENSE_FILE'] = os.environ["HOME"] + "/gurobi_license/gurobi.lic"

# %%
LOAD_FACTOR  = 1
TEST_CONFIG = [
    {"line_list": [], "switch_list":  []},
    {"line_list": [6, 9], "switch_list":  [25, 28]},
    {"line_list": [2, 6, 9], "switch_list":  [21, 25, 28]},
    {"line_list": [16], "switch_list":  [35]},
    {"line_list": [1], "switch_list":  [20]},
    {"line_list": [10], "switch_list":  [29]},
    {"line_list": [7, 11], "switch_list":  [26, 30]}
]
NB_TEST = 0
# set input data

net = pp.from_pickle(".cache/input_data/mv_example.p")
combination_results = pl.read_csv(".cache/input_data/load_facotr_1_results.csv")\
    .with_columns(
        pl.concat_list(cs.starts_with("switch")).alias("switch_list")
    )

net["load"]["p_mw"] = net["load"]["p_mw"]*LOAD_FACTOR
net["load"]["q_mvar"] = net["load"]["q_mvar"]*LOAD_FACTOR


net["line"].loc[:, "max_i_ka"] = 1
net["line"].loc[TEST_CONFIG[NB_TEST]["line_list"], "max_i_ka"] = 1e-2
# %%
base_grid_data = pandapower_to_dig_a_plan_schema(net)
dig_a_plan: DigAPlan = DigAPlan(verbose= True, big_m = 1e2)

dig_a_plan.add_grid_data(**base_grid_data)
dig_a_plan.solve_models_pipeline(max_iters = 1)

# %%
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.01, row_titles = ['Slave objective', 'Master objective'])
fig.add_trace(go.Scatter(go.Scatter(y=dig_a_plan.slave_obj_list[1:]), mode='lines', name='Slave objective'), row=1, col=1)
fig.add_trace(go.Scatter(go.Scatter(y=dig_a_plan.master_obj_list[1:]), mode='lines', name='Master objective'), row=2, col=1)
fig.update_layout(height= 400, width=600, margin=dict(t=10, l=20, r= 10, b=10))

fig.show()

# %%
# node_data, edge_data =  compare_dig_a_plan_with_pandapower(dig_a_plan=dig_a_plan, net=net)
plot_grid_from_pandapower(net=net, dig_a_plan=dig_a_plan)

# %%

dig_a_plan.master_obj_list
# %%

dig_a_plan.slave_obj
# %%
print(dig_a_plan.marginal_cost.to_pandas().to_string())
# %%
