# %% ------------------ setup & imports ------------------
import os
os.chdir(os.getcwd().replace("/src", ""))

from examples import *   

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Pyomo for objective evaluation
try:
    import pyomo.environ as pe
except Exception:
    pe = None

# ---------------- helpers: get 1 objective per scenario from ADMM ----------------
def _get_active_objective_safe(model):
    """
    Return a constructed Objective on a Pyomo concrete model, else None.
    Tries common names, then any active constructed Objective.
    """
    if pe is None:
        return None
    for name in ("obj", "objective", "Objective"):
        if hasattr(model, name):
            obj = getattr(model, name)
            try:
                if isinstance(obj, pe.Objective) and obj.is_constructed():
                    return obj
            except Exception:
                pass
    try:
        for obj in model.component_objects(pe.Objective, active=True):
            if getattr(obj, "is_constructed", lambda: False)():
                return obj
    except Exception:
        pass
    return None

def collect_objectives_from_admm_instances(dap):
    """
    Use ADMM's per-scenario model instances to get one objective per scenario.
    This avoids double-counting subproblems found by auto-discovery.
    """
    mm = getattr(dap, "model_manager", None)
    if mm is None or not hasattr(mm, "admm_model_instances"):
        raise RuntimeError("model_manager.admm_model_instances not found.")
    rows = []
    for omega, model in mm.admm_model_instances.items():
        obj = _get_active_objective_safe(model)
        if obj is None:
            continue
        rows.append({"scenario": int(omega), "objective": float(pe.value(obj))}) #type:ignore
    if not rows:
        raise RuntimeError("No constructed Objectives found in admm_model_instances.")
    return pd.DataFrame(rows).sort_values("scenario").reset_index(drop=True)

# ---------------- plotting helpers ----------------
def plot_objective_density_from_df(df_obj, out_svg_path):
    vals = df_obj["objective"].to_numpy(dtype=float)
    if vals.size == 0:
        print("No objective values to plot.")
        return
    os.makedirs(os.path.dirname(out_svg_path), exist_ok=True)
    plt.figure(figsize=(9, 5))
    plt.hist(vals, bins=20, density=True, alpha=0.6)
    xs = np.linspace(vals.min(), vals.max(), 256)
    def kde(samples, x, bw=None):
        samples = np.asarray(samples, float)
        if bw is None:
            std = samples.std(ddof=1) if len(samples) > 1 else 1.0
            bw = 1.06 * std * (len(samples) ** (-1/5)) if std > 0 else 1.0
        return (1/(len(samples)*bw*np.sqrt(2*np.pi)) *
                np.sum(np.exp(-0.5*((x[:,None]-samples[None,:])/bw)**2), axis=1))
    plt.plot(xs, kde(vals, xs), linewidth=2)
    plt.title("Scenario-wise Objective Density")
    plt.xlabel("Objective value")
    plt.ylabel("Probability density")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_svg_path, bbox_inches="tight")
    plt.show()

def plot_objective_vs_scenario(df_obj, out_svg_path):
    df_plot = df_obj.copy()
    df_plot["scenario"] = df_plot["scenario"].astype(int)
    df_plot = df_plot.sort_values("scenario").reset_index(drop=True)
    plt.figure(figsize=(12, 5))
    plt.plot(df_plot["scenario"], df_plot["objective"], marker="o", linewidth=1)
    plt.xlabel("Scenario")
    plt.ylabel("Objective value")
    plt.title("Objective per Scenario")
    plt.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(out_svg_path), exist_ok=True)
    plt.savefig(out_svg_path, bbox_inches="tight")
    plt.show()

def plot_objective_by_rank(df_obj, out_svg_path):
    df_sorted = df_obj.sort_values("objective").reset_index(drop=True)
    df_sorted["rank"] = np.arange(1, len(df_sorted) + 1)
    plt.figure(figsize=(12, 5))
    plt.plot(df_sorted["rank"], df_sorted["objective"], marker="o", linewidth=1)
    plt.xlabel("Rank (1 = lowest objective)")
    plt.ylabel("Objective value")
    plt.title("Objective by Rank")
    plt.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(out_svg_path), exist_ok=True)
    plt.savefig(out_svg_path, bbox_inches="tight")
    plt.show()

# %% ------------------ data & groups ------------------
net = pp.from_pickle("data/simple_grid.p")
grid_data = pandapower_to_dig_a_plan_schema_with_scenarios(
    net,
    number_of_random_scenarios=100,   # <-- expected number of scenarios
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

# %% ------------------ configure & build ADMM ------------------
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

# %% ------------------ run ADMM ------------------
dap.solve_model()  

# Quick sanity: how many scenarios are actually present?
print("Scenarios in data_manager:", len(dap.data_manager.grid_data_parameters_dict)) #type:ignore
print("Scenarios in Ω:", len(getattr(dap.model_manager, "Ω", [])))
print("ADMM model instances:", len(getattr(dap.model_manager, "admm_model_instances", {})))

# %% ------------- collect scenario objectives (ONE per scenario) -------------
os.makedirs(".cache/figs", exist_ok=True)
df_obj = collect_objectives_from_admm_instances(dap)
print("Collected objectives (ADMM):", len(df_obj))
df_obj.to_csv(".cache/figs/objectives_by_scenario_ADMM.csv", index=False)

# --------- PLOTS ----------
plot_objective_density_from_df(df_obj, out_svg_path=".cache/figs/objective_density_by_scenario_ADMM.svg")
plot_objective_vs_scenario(df_obj, out_svg_path=".cache/figs/objective_per_scenario_ADMM.svg")
plot_objective_by_rank(df_obj, out_svg_path=".cache/figs/objective_by_rank_ADMM.svg")

# %% ------------------ Normal Open baseline & comparison ------------------
# Build baseline (fixed switches to normal_open)
dap_fixed = copy.deepcopy(dap)
dap_fixed.solve_model(fixed_switches=True)

# Collect baseline objectives (ONE per scenario)
df_no = collect_objectives_from_admm_instances(dap_fixed).rename(
    columns={"objective": "objective_NormalOpen"}
)
df_no["scenario"] = df_no["scenario"].astype(int)

# Merge with ADMM
df_admm = df_obj.rename(columns={"objective": "objective_ADMM"})
df_admm["scenario"] = df_admm["scenario"].astype(int)

df_cmp = pd.merge(df_admm, df_no, on="scenario", how="inner").sort_values("scenario").reset_index(drop=True)
df_cmp["delta"] = df_cmp["objective_ADMM"] - df_cmp["objective_NormalOpen"]
df_cmp.to_csv(".cache/figs/objectives_ADMM_vs_NormalOpen.csv", index=False)
print("Scenarios in comparison:", len(df_cmp))
print(df_cmp.head())

# Overlay: objective vs scenario
plt.figure(figsize=(12, 5))
plt.plot(df_cmp["scenario"], df_cmp["objective_ADMM"], marker="o", linewidth=1, label="ADMM")
plt.plot(df_cmp["scenario"], df_cmp["objective_NormalOpen"], marker="s", linewidth=1, label="Normal Open")
plt.xlabel("Scenario")
plt.ylabel("Objective value")
plt.title("Objective per Scenario: ADMM vs Normal Open")
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig(".cache/figs/objective_per_scenario_ADMM_vs_NormalOpen.svg", bbox_inches="tight")
plt.show()

# Delta: (ADMM − Normal Open) vs scenario
plt.figure(figsize=(12, 5))
plt.axhline(0.0, linestyle="--", linewidth=1)
plt.plot(df_cmp["scenario"], df_cmp["delta"], marker="o", linewidth=1)
plt.xlabel("Scenario")
plt.ylabel("Δ Objective (ADMM − Normal Open)")
plt.title("Objective Difference per Scenario (ADMM − Normal Open)")
plt.grid(True, alpha=0.3)
plt.savefig(".cache/figs/objective_delta_per_scenario_ADMM_minus_NormalOpen.svg", bbox_inches="tight")
plt.show()

# Density comparison
def _kde_1d(samples, x, bw=None):
    samples = np.asarray(samples, float)
    if bw is None:
        std = samples.std(ddof=1) if len(samples) > 1 else 1.0
        bw = 1.06 * std * (len(samples) ** (-1/5)) if std > 0 else 1.0
    return (1/(len(samples)*bw*np.sqrt(2*np.pi)) *
            np.sum(np.exp(-0.5*((x[:,None]-samples[None,:])/bw)**2), axis=1))

vals_admm = df_cmp["objective_ADMM"].to_numpy(float)
vals_no   = df_cmp["objective_NormalOpen"].to_numpy(float)
xmin = float(min(vals_admm.min(), vals_no.min()))
xmax = float(max(vals_admm.max(), vals_no.max()))
xs = np.linspace(xmin, xmax, 256)

plt.figure(figsize=(10,5))
plt.plot(xs, _kde_1d(vals_admm, xs), linewidth=2, label="ADMM")
plt.plot(xs, _kde_1d(vals_no, xs),   linewidth=2, label="Normal Open")
plt.xlabel("Objective value")
plt.ylabel("Probability density")
plt.title("Objective Distribution: ADMM vs Normal Open")
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig(".cache/figs/objective_density_ADMM_vs_NormalOpen.svg", bbox_inches="tight")
plt.show()

# %% ------------------ optional: other plots ------------------
# Consensus switch/tap states (from ADMM)
print(dap.model_manager.zδ_variable)
print(dap.model_manager.zζ_variable)

# Compare with pandapower & plot annotated grid (ADMM)
node_data, edge_data = compare_dig_a_plan_with_pandapower(dig_a_plan=dap, net=net, from_z=True)
fig = plot_grid_from_pandapower(net=net, dap=dap, from_z=True)

# Distribution plots: ADMM vs Normal Open
nodal_variables = ["voltage"]
edge_variables = ["current", "p_flow", "q_flow"]
for variable in nodal_variables + edge_variables:
    plot_distribution_variable(
        daps={"ADMM": dap, "Normal Open": dap_fixed},
        variable_name=variable,
        variable_type=("nodal" if variable in nodal_variables else "edge"),
    )

# Convergence (r_norm & s_norm)
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
plt.savefig(".cache/figs/admm_convergence.svg", bbox_inches="tight")
plt.show()

# %% ------------------ optional: other plots ------------------
# Box & whiskers: ADMM vs Normal Open
data = [
    df_cmp["objective_ADMM"].to_numpy(float),
    df_cmp["objective_NormalOpen"].to_numpy(float),
]

plt.figure(figsize=(7, 5))
bp = plt.boxplot(
    data,
    labels=["ADMM", "Normal Open"], #type:ignore
    patch_artist=True,  
    showmeans=True,      
)

# Optional: some light styling (no specific colors requested)
for box in bp["boxes"]:
    box.set_alpha(0.5)
for whisk in bp["whiskers"]:
    whisk.set_linewidth(1.5)
for cap in bp["caps"]:
    cap.set_linewidth(1.5)
for median in bp["medians"]:
    median.set_linewidth(2.0)
for mean in bp["means"]:
    mean.set_marker("x")
    mean.set_markersize(8)

plt.ylabel("Objective value")
plt.title("Objective Distribution Across Scenarios")
plt.grid(True, axis="y", alpha=0.3)
os.makedirs(".cache/figs", exist_ok=True)
plt.savefig(".cache/figs/boxplot_admm_vs_normalopen.svg", bbox_inches="tight")
plt.show()

