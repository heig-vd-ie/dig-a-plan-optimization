# %%
import os

# %% import libraries
from pathlib import Path
from experiments import *

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# %% set parameters
net = pp.from_pickle(str(PROJECT_ROOT / "examples/ieee-33/simple_grid.p"))
base_grid_data = pandapower_to_dig_a_plan_schema_with_scenarios(
    net, number_of_random_scenarios=12
)

# %% initialize DigAPlan

konfig = CombinedConfig(
    verbose=True,
    big_m=1e3,
    ε=1,
    γ_infeasibility=100.0,
    γ_admm_penalty=0.0,
    all_scenarios=True,
    time_limit=1800,
)
dig_a_plan = DigAPlanCombined(konfig=konfig)

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

