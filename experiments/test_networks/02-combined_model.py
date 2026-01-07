# %%
import os
print(os.environ.get("GRB_LICENSE_FILE"))

# %% import libraries
from experiments import *

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]  


# %% set parameters
net = pp.from_pickle(str(PROJECT_ROOT / "examples/ieee-33/simple_grid.p"))
base_grid_data = pandapower_to_dig_a_plan_schema_with_scenarios(net)

# %% initialize DigAPlan


konfig = CombinedConfig(
    verbose=True,      
    threads=1,
    big_m=1e2,
    γ_infeasibility=1.0,
    factor_v=1,
    factor_i=1e-3,
)

dig_a_plan = DigAPlanCombined(konfig=konfig)

# %% add grid data and solve the combined model
dig_a_plan.add_grid_data(base_grid_data)
dig_a_plan.solve_model()  # has error in solver

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

# %% plot the grid annotated with DigAPlan results
fig = plot_grid_from_pandapower(dap=dig_a_plan)

# %% compare DigAPlan results with pandapower results

node_data, edge_data = compare_dig_a_plan_with_pandapower(
    dig_a_plan=dig_a_plan, net=net
)
