# %% import libraries
import os
import pandapower as pp
import plotly.graph_objs as go

from data_connector import pandapower_to_dig_a_plan_schema
from data_display.grid_plotting import plot_grid_from_pandapower
from data_display.output_processing import compare_dig_a_plan_with_pandapower
from pipelines.combined_dig_a_plan import DigAPlan  

from pyomo_utility import extract_optimization_results
from plotly.subplots import make_subplots

# ensure working directory is project root
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
dig_a_plan = DigAPlan(
    verbose=False,
    big_m=1e2,          # your big‐M value
    power_factor=1e-3,  # scaling for p_flow/q_flow
    current_factor=1e-3,
    voltage_factor=1.0,
)

# %% add grid data and solve the combined model
dig_a_plan.add_grid_data(**base_grid_data)
dig_a_plan.solve_combined_model()  # one‐shot solve

