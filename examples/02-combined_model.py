# %%
import os

os.chdir(os.getcwd().replace("/src", ""))
# %% import libraries
from examples import *

# %% set parameters
net = pp.from_pickle("data/simple_grid.p")
base_grid_data = pandapower_to_dig_a_plan_schema(
    net, taps=[95, 98, 99, 100, 101, 102, 105]
)

# %% initialize DigAPlan

config = CombinedConfig(
    verbose=True,
    big_m=1e3,
    ε=1,
    pipeline_type=PipelineType.COMBINED,
    γ_infeasibility=1.0,
    γ_admm_penalty=0.0,
)
dig_a_plan = DigAPlanCombined(config=config)

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
# Power flow
powers = dig_a_plan.result_manager.extract_edge_active_power_flow()
reactive_powers = dig_a_plan.result_manager.extract_edge_reactive_power_flow()
taps = dig_a_plan.result_manager.extract_transformer_tap_position()
print(taps)

# %% plot the grid annotated with DigAPlan results
fig = plot_grid_from_pandapower(net=net, dap=dig_a_plan)

# %% compare DigAPlan results with pandapower results

node_data, edge_data = compare_dig_a_plan_with_pandapower(
    dig_a_plan=dig_a_plan, net=net
)
