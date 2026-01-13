# %%
from experiments import *
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# %% Convert pandapower -> DigAPlan schema with a few scenarios
seed = 42
pp_path = PROJECT_ROOT / ".cache" / "input" / "boisy" / "boisy_grid.p"

grid = GridCaseModel(
    pp_file=str(pp_path),
    s_base=1e6,
)
stu = ShortTermUncertaintyRandom(
    n_scenarios=10,
    v_bounds=(-0.07, 0.07),
    p_bounds=(-0.5, 1.0),
    q_bounds=(-0.5, 0.5),
)

# %% --- CLEAN NULLS IN THE RAW PANDAPOWER NET (same as your 2nd script) ---
net, grid_data = get_grid_case(grid=grid, seed=seed, stu=stu)

# %% convert pandapower grid to DigAPlan grid data

grid_data.edge_data = grid_data.edge_data.with_columns(
    pl.when(c(col) < 1e-3).then(pl.lit(0)).otherwise(c(col)).alias(col)
    for col in ["b_pu", "r_pu", "x_pu"]
).with_columns(
    c("normal_open").fill_null(False),
)

# %% Configure ADMM pipeline
konfig = ADMMConfig(
    verbose=False,
    solver_name="gurobi",
    solver_non_convex=0,  # Set non-convex parameters to 2 for Boisy grid
    big_m=1e3,
    ε=1e-4,
    ρ=2.0,
    γ_infeasibility=100.0,
    γ_admm_penalty=1.0,
    time_limit=1,  # TODO: set time limit to 10 seconds for actual boisy grid
    max_iters=10,
    μ=10.0,
    τ_incr=2.0,
    τ_decr=2.0,
    mutation_factor=2,
    groups=10,  # TODO: set number of groups for actual boisy grid to 40
)

dap = DigAPlanADMM(konfig=konfig)

# %% Build per-scenario models (instantiated inside add_grid_data)
dap.add_grid_data(grid_data)


# %% Run ADMM
dap.model_manager.solve_model()

# %%
dap_fixed = copy.deepcopy(dap)
dap_fixed.solve_model(fixed_switches=True)

# %% Inspect consensus and per-scenario deltas

print("\n=== ADMM consensus switch states (z) ===")
print(dap.model_manager.zδ_variable)
print(dap.model_manager.zζ_variable)

# %%
save_dap_state(dap, ".cache/figs/boisy_dap")
save_dap_state(dap_fixed, ".cache/figs/boisy_dap_fixed")
joblib.dump(net, ".cache/figs/boisy_net.joblib")

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
        daps={"ADMM": dap, "Normal Open": dap_fixed},  # type: ignore
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
plt.show()

# %%
plot_grid_from_pandapower(net=net, dap=dap, from_z=True, color_by_results=True, node_size=6)  # type: ignore

# %% Plot fixed switches
plot_grid_from_pandapower(net=net, dap=dap_fixed, from_z=True, color_by_results=True, node_size=6)  # type: ignore
