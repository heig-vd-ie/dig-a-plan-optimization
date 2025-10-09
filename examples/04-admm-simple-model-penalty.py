# %%
import os

os.chdir(os.getcwd().replace("/src", ""))

# %%
from examples import *
import copy

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

# quick grid stats
print("Number of Nodes:", len(net.bus))
print("Number of Lines:", len(net.line))
print("Number of Switches:", len(net.switch))
print("Number of Transformers:", len(net.trafo))
print("Number of Loads:", len(net.load))
print(
    "Total PVs:",
    (net.sgen.type == "pv").sum() if "type" in net.sgen.columns else len(net.sgen),
)
print("Total Nominal Load (MW):", net.load.p_mw.sum())
print(
    "Total Nominal PVs (MW):",
    (
        net.sgen.loc[net.sgen.type == "pv", "p_mw"].sum()
        if "type" in net.sgen.columns
        else net.sgen.p_mw.sum()
    ),
)

# %% Configure ADMM pipeline
config = ADMMConfig(
    verbose=False,
    pipeline_type=PipelineType.ADMM,
    solver_name="gurobi",
    solver_non_convex=2,
    big_m=1e3,
    ε=1,  # this is ε_primal/ε_dual
    ρ=2.0,  # <--- initial rho
    γ_infeasibility=10,
    γ_admm_penalty=1.0,
    γ_trafo_loss=1e2,
    groups=groups,
    max_iters=200,  # give room for convergence
    μ=20.0,  # <--- change here
    τ_incr=3.0,  # <--- change here
    τ_decr=3.0,  # <--- change here
)

dap = DigAPlanADMM(config=config)
dap.add_grid_data(grid_data)

# %% Run ADMM
dap.model_manager.solve_model()

# %% === ADD: convergence iteration and time printout ===
import numpy as np

eps_pr = config.ε_primal
eps_du = config.ε_dual

r = np.asarray(dap.model_manager.r_norm_list, dtype=float)
s = np.asarray(dap.model_manager.s_norm_list, dtype=float)
t = np.asarray(dap.model_manager.time_list, dtype=float)

# Your plot uses time_list[1:] vs residual lists, so residual i -> time index i+1
idx = np.where((r <= eps_pr) & (s <= eps_du))[0]

if idx.size > 0 and t.size >= (idx[0] + 2):
    k_converged = int(idx[0]) + 1  # 1-based iteration count
    t_converged = float(t[idx[0] + 1] - t[0])  # seconds from start
    print(
        f"[ADMM] Converged at iteration {k_converged} in {t_converged:.2f} s "
        f"(eps_pr={eps_pr}, eps_du={eps_du})."
    )
else:
    k_converged = None
    t_converged = None
    print("[ADMM] Did not meet both tolerances within the run.")

# also report final residuals and current rho (after any adaptation)
r_final = float(r[-1]) if r.size else float("nan")
s_final = float(s[-1]) if s.size else float("nan")
print(
    f"[ADMM] Final residuals: r={r_final:.3e}, s={s_final:.3e}; rho(now)={config.ρ:.6g}"
)
