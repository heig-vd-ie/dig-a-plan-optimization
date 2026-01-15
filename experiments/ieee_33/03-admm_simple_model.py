# %%
from experiments import *
from pathlib import Path
from api.grid_cases import get_grid_case
from data_model.kace import GridCaseModel
from data_model.reconfiguration import ShortTermUncertaintyRandom

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# %% --- Load net via API  ---
grid = GridCaseModel(
    pp_file=str(PROJECT_ROOT / "examples" / "ieee_33" / "simple_grid.p"),
    s_base=1e6,
)
stu = ShortTermUncertaintyRandom()

net, grid_data = get_grid_case(grid=grid, seed=42, stu=stu)
# %% --- build grid data with scenarios ---
net.bus["max_vm_pu"] = 1.05
net.bus["min_vm_pu"] = 0.95
groups = {
    0: [19, 20, 21, 29, 32, 35],
    1: [35, 30, 33, 25, 26, 27],
    2: [27, 32, 22, 23, 34],
    3: [31, 24, 28, 21, 22, 23],
    4: [34, 26, 25, 24, 31],
}


# %% Configure ADMM pipeline
konfig = ADMMConfig(
    verbose=False,
    solver_name="gurobi",
    solver_non_convex=2,
    big_m=1e3,
    ε=1,
    ρ=2.0,
    γ_infeasibility=10,
    γ_admm_penalty=1.0,
    γ_trafo_loss=1e2,
    groups=groups,
    max_iters=20,
    μ=10.0,
    τ_incr=2.0,
    τ_decr=2.0,
)

dap = DigAPlanADMM(konfig=konfig)
dap.add_grid_data(grid_data)


# %% Run ADMM
dap.model_manager.solve_model()
# %% Consensus switch states (one value per switch)
print(dap.model_manager.zδ_variable)
print(dap.model_manager.zζ_variable)
# %% compare DigAPlan results with pandapower results
node_data, edge_data = compare_dig_a_plan_with_pandapower(
    dig_a_plan=dap, net=net, from_z=True
)
# %% plot the grid annotated with DigAPlan results
fig = plot_grid_from_pandapower(dap=dap, from_z=True)

# %% Fixed switches
dap_fixed = copy.deepcopy(dap)
dap_fixed.solve_model(fixed_switches=True)

# %% Plot Distribution
nodal_variables = [
    "voltage",
    # "p_curt_cons",
    # "p_curt_prod",
    # "q_curt_cons",
    # "q_curt_prod",
]
edge_variables = [
    "current",
    "p_flow",
    "q_flow",
]
for variable in nodal_variables + edge_variables:
    DistributionVariable(
        daps={"ADMM": dap, "Normal Open": dap_fixed},
        variable_name=variable,
        variable_type=("nodal" if variable in nodal_variables else "edge"),
    ).plot()

# %% Plot iteration of r_norm and s_norm
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(12, 6))
plt.plot(
    np.array(dap.model_manager.time_list[1:]) - dap.model_manager.time_list[0],
    dap.model_manager.r_norm_list,
    label="r_norm",
    marker="o",
)
plt.plot(
    np.array(dap.model_manager.time_list[1:]) - dap.model_manager.time_list[0],
    dap.model_manager.s_norm_list,
    label="s_norm",
    marker="o",
)
plt.xlabel("Seconds")
plt.ylabel("Norm Value")
plt.title("ADMM Iteration: r_norm and s_norm")
plt.legend()
plt.grid()
plt.savefig(".cache/figs/admm_convergence.svg")
plt.show()
