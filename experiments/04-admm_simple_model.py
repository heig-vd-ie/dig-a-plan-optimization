# %%
import os

os.chdir(os.getcwd().replace("/src", ""))
# %%
from experiments import *

# %% set parameters

net = pp.from_pickle("data/simple_grid.p")
grid_data = pandapower_to_dig_a_plan_schema_with_scenarios(
    net,
    number_of_random_scenarios=100,
    p_bounds=(-0.6, 1.5),
    q_bounds=(-0.1, 0.1),
    v_bounds=(-0.1, 0.1),
    v_min=0.95,
    v_max=1.05,
)
groups = {
    0: [19, 20, 21, 29, 32, 35],
    1: [35, 30, 33, 25, 26, 27],
    2: [27, 32, 22, 23, 34],
    3: [31, 24, 28, 21, 22, 23],
    4: [34, 26, 25, 24, 31],
}


# %% Configure ADMM pipeline
config = ADMMConfig(
    verbose=False,
    pipeline_type=PipelineType.ADMM,
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

dap = DigAPlanADMM(config=config)
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
fig = plot_grid_from_pandapower(net=net, dap=dap, from_z=True)

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
    plot_distribution_variable(
        daps={"ADMM": dap, "Normal Open": dap_fixed},
        variable_name=variable,
        variable_type=("nodal" if variable in nodal_variables else "edge"),
    )

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
