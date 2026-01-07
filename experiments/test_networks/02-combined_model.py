# %% import libraries
from experiments import *
from pathlib import Path

from api.grid_cases import get_grid_case
from data_model.kace import GridCaseModel
from data_model.reconfiguration import ShortTermUncertainty
PROJECT_ROOT = Path(__file__).resolve().parents[2]  


# %% --- load net via API
grid = GridCaseModel(
    pp_file=str(PROJECT_ROOT / "examples" / "ieee-33" / "simple_grid.p"),
    s_base=1e6,
    cosφ=0.95,
)

stu = ShortTermUncertainty(
    number_of_scenarios=10,
    p_bounds=(-0.2, 0.2),
    q_bounds=(-0.2, 0.2),
    v_bounds=(-0.03, 0.03),
)

net, _ = get_grid_case(grid=grid, seed=42, stu=stu)

# %% --- build base_grid_data ---
base_grid_data = pandapower_to_dig_a_plan_schema_with_scenarios(
    net=net,
    s_base=grid.s_base,
    number_of_random_scenarios=stu.number_of_scenarios,
    p_bounds=stu.p_bounds,
    q_bounds=stu.q_bounds,
    v_bounds=stu.v_bounds,
    seed=42,
)

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
