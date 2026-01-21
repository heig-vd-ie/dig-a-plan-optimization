# %% import libraries
from pathlib import Path
from experiments import *
from api.grid_cases import get_grid_case
from data_model.kace import GridCaseModel
from data_model.reconfiguration import ShortTermUncertaintyRandom

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# %% set parameters from API

grid = GridCaseModel(
    pp_file=str(PROJECT_ROOT / "examples" / "ieee-33" / "simple_grid.p"),
    s_base=1e6,
)
stu = ShortTermUncertaintyRandom(
    n_scenarios=12,
)
net, base_grid_data = get_grid_case(grid=grid, seed=42, stu=stu)


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
