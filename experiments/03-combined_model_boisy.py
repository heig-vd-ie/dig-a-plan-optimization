# %%
import os

os.chdir(os.getcwd().replace("/src", ""))
# %% import libraries
from experiments import *

# %% set parameters
if USE_SIMPLIFIED_GRID := True:
    net = pp.from_pickle(".cache/boisy_grid_simplified.p")
    base_grid_data = pandapower_to_dig_a_plan_schema_with_scenarios(net)
else:
    net = pp.from_pickle(".cache/boisy_grid.p")
    base_grid_data = pandapower_to_dig_a_plan_schema_with_scenarios(net)


# %% convert pandapower grid to DigAPlan grid data

base_grid_data.edge_data = base_grid_data.edge_data.with_columns(
    pl.when(c(col) < 1e-3).then(pl.lit(0)).otherwise(c(col)).alias(col)
    for col in ["b_pu", "r_pu", "x_pu"]
).with_columns(
    c("normal_open").fill_null(False),
)
# %% initialize DigAPlan

config = CombinedConfig(
    verbose=True,
    big_m=1000,
    ε=0.1,
    pipeline_type=PipelineType.COMBINED,
    γ_admm_penalty=0.0,
)
dig_a_plan = DigAPlanCombined(config=config)

# %% add grid data and solve the combined model
dig_a_plan.add_grid_data(base_grid_data)
dig_a_plan.solve_model(groups=5)  # one‐shot solve

# %% extract and compare results
# Switch status
switches = dig_a_plan.result_manager.extract_switch_status()
# Node voltages
voltages = dig_a_plan.result_manager.extract_node_voltage()
# Line currents
currents = dig_a_plan.result_manager.extract_edge_current()
active_power_flow = dig_a_plan.result_manager.extract_edge_active_power_flow()
reactive_power_flow = dig_a_plan.result_manager.extract_edge_reactive_power_flow()
