# %%
from pathlib import Path
from experiments import *


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# %% set parameters
seed = 42
pp_path = PROJECT_ROOT / ".cache" / "input" / "boisy" / "boisy_grid.p"

grid = GridCaseModel(
    pp_file=str(pp_path),
    s_base=1e6,
)
stu = ShortTermUncertaintyRandom()
# %% convert pandapower grid to DigAPlan grid data
net, base_grid_data = get_grid_case(grid=grid, seed=seed, stu=stu)
base_grid_data.edge_data = base_grid_data.edge_data.with_columns(
    pl.when(c(col) < 1e-3).then(pl.lit(0)).otherwise(c(col)).alias(col)
    for col in ["b_pu", "r_pu", "x_pu"]
).with_columns(
    c("normal_open").fill_null(False),
)
# %% initialize DigAPlan

konfig = CombinedConfig(
    verbose=True,
    big_m=1000,
    ε=0.1,
    γ_admm_penalty=0.0,
)
dig_a_plan = DigAPlanCombined(konfig=konfig)

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
