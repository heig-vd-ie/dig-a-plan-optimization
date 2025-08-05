# %% import libraries
import os
import pandapower as pp

from data_exporter.pandapower_to_dig_a_plan import pandapower_to_dig_a_plan_schema
from data_display.grid_plotting import plot_grid_from_pandapower
from data_display.output_processing import compare_dig_a_plan_with_pandapower
from pipelines import DigAPlan
from pipelines.configs import CombinedConfig, PipelineType
from pipelines.model_managers.bender import PipelineModelManagerBender


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
base_grid_data = pandapower_to_dig_a_plan_schema(net, number_of_random_scenarios=10)

# %% initialize DigAPlan

config = CombinedConfig(
    verbose=True,
    big_m=1e3,
    ε=1,
    pipeline_type=PipelineType.COMBINED,
    γ_infeasibility=100.0,
    γ_admm_penalty=0.0,
    all_scenarios=True,
)
dig_a_plan = DigAPlan(config=config)

# %% add grid data and solve the combined model
dig_a_plan.add_grid_data(base_grid_data)
dig_a_plan.solve_model()  # one‐shot solve

# %% extract and compare results
# Switch status
switches = dig_a_plan.result_manager.extract_switch_status()
# Node voltages
voltages = dig_a_plan.result_manager.extract_node_voltage()
# Line currents
currents = dig_a_plan.result_manager.extract_edge_current()


# %% plot the grid annotated with DigAPlan results
fig = plot_grid_from_pandapower(net, dig_a_plan)

# %% compare DigAPlan results with pandapower results
if isinstance(dig_a_plan.model_manager, PipelineModelManagerBender):
    raise ValueError(
        "The model manager is not a Combined model manager, but a Bender model manager."
    )
node_data, edge_data = compare_dig_a_plan_with_pandapower(
    dig_a_plan=dig_a_plan, net=net
)
