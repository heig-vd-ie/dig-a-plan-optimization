# %% ------------------ setup & imports ------------------
import os
os.chdir(os.getcwd().replace("/src", ""))

from examples import *   

import copy
import types
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Pyomo (for model detection and objective evaluation)
try:
    import pyomo.environ as pe
except Exception:
    pe = None


# %% -------- SAFE discovery: Concrete models & constructed Objectives --------
from collections import deque

def _is_concrete_model(obj):
    """True only for constructed ConcreteModel / concrete Blocks."""
    if pe is None:
        return False
    try:
        if isinstance(obj, pe.ConcreteModel):
            return True
        if isinstance(obj, pe.Block) and obj.is_constructed():
            return True
    except Exception:
        pass
    return False

def _get_active_objective_safe(model):
    """Return a constructed Objective on a concrete model, else None."""
    if pe is None:
        return None
    # try common names first
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

def _safe_iter_dict(d):
    try:
        return list(d.items())
    except Exception:
        return []

def _safe_iter_seq(seq):
    try:
        return list(enumerate(seq))
    except Exception:
        return []

def discover_concrete_models(root, max_depth=5, max_items=8000):
    """
    BFS to find only Concrete Pyomo models/blocks reachable from root.
    Returns list of (path, model).
    """
    seen = set()
    q = deque()
    q.append(("", root, 0))
    found = []
    visits = 0

    while q and visits < max_items:
        path, obj, depth = q.popleft()
        visits += 1
        oid = id(obj)
        if oid in seen:
            continue
        seen.add(oid)

        if _is_concrete_model(obj):
            found.append((path or "<root>", obj))
            continue

        if depth >= max_depth:
            continue

        if isinstance(obj, dict):
            for k, v in _safe_iter_dict(obj):
                q.append((f"{path}.{k}" if path else str(k), v, depth + 1))
            continue

        if isinstance(obj, (list, tuple)):
            for i, v in _safe_iter_seq(obj):
                q.append((f"{path}[{i}]" if path else f"[{i}]", v, depth + 1))
            continue

        if isinstance(obj, (types.ModuleType, types.FunctionType, type)):
            continue
        if pe is not None and isinstance(obj, (pe.Component, pe.ComponentUID)) and not _is_concrete_model(obj):
            continue

        try:
            for attr in dir(obj):
                if attr.startswith("__"):
                    continue
                try:
                    val = getattr(obj, attr)
                except Exception:
                    continue
                q.append((f"{path}.{attr}" if path else attr, val, depth + 1))
        except Exception:
            pass

    return found

def collect_objectives_via_discovery(dap):
    """
    1) Try known locations (model_manager.models / .scenario_models) if they are Concrete.
    2) Else discover only Concrete models and evaluate only constructed Objectives.

    Returns: DataFrame with columns: scenario, objective, path
    """
    rows = []

    def _try_models_dict(md, tag):
        ok = False
        for s, m in sorted(md.items(), key=lambda kv: kv[0]):
            if not _is_concrete_model(m):
                continue
            obj = _get_active_objective_safe(m)
            if obj is None:
                continue
            try:
                val = pe.value(obj)
                rows.append({"scenario": int(s), "objective": float(val), "path": tag})
                ok = True
            except Exception:
                continue
        return ok

    mm = getattr(dap, "model_manager", None)
    if mm is not None:
        models = getattr(mm, "models", None)
        if isinstance(models, dict) and models:
            if _try_models_dict(models, "model_manager.models"):
                return pd.DataFrame(rows)
        elif isinstance(models, (list, tuple)) and models:
            for s, m in enumerate(models):
                if not _is_concrete_model(m):
                    continue
                obj = _get_active_objective_safe(m)
                if obj is None:
                    continue
                try:
                    val = pe.value(obj)
                    rows.append({"scenario": int(s), "objective": float(val), "path": "model_manager.models(list)"})
                except Exception:
                    continue
            if rows:
                return pd.DataFrame(rows)
        else:
            scen_models = getattr(mm, "scenario_models", None)
            if isinstance(scen_models, dict) and scen_models:
                if _try_models_dict(scen_models, "model_manager.scenario_models"):
                    return pd.DataFrame(rows)

    # 2) Discover only concrete models
    candidates = []
    for anchor_name in ("dap", "dap.model_manager"):
        anchor = dap if anchor_name == "dap" else getattr(dap, "model_manager", None)
        if anchor is None:
            continue
        for pth, mdl in discover_concrete_models(anchor, max_depth=5):
            candidates.append((f"{anchor_name}:{pth}", mdl))

    rows = []
    for s, (pth, m) in enumerate(candidates):
        obj = _get_active_objective_safe(m)
        if obj is None:
            continue
        try:
            val = pe.value(obj)
            rows.append({"scenario": s, "objective": float(val), "path": pth})
        except Exception:
            continue

    if not rows:
        raise RuntimeError("No constructed Objectives found on any ConcreteModel reachable from dap.")

    df = pd.DataFrame(rows).sort_values("scenario").reset_index(drop=True)
    print("Discovered Concrete Pyomo models & objectives at (first 20):")
    print(df[["scenario", "path"]].head(20))
    return df


# %% ------------------ plotting helpers ------------------
def plot_objective_density_from_df(df_obj, out_svg_path=".cache/figs/objective_density_by_scenario.svg"):
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

def plot_objective_vs_scenario(df_obj, out_svg_path=".cache/figs/objective_per_scenario.svg"):
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

def plot_objective_vs_scenario_smoothed(df_obj, bandwidth=5.0, out_svg_path=".cache/figs/objective_per_scenario_smoothed.svg"):
    df_plot = df_obj.copy()
    df_plot["scenario"] = df_plot["scenario"].astype(int)
    df_plot = df_plot.sort_values("scenario").reset_index(drop=True)
    x = df_plot["scenario"].to_numpy()
    y = df_plot["objective"].to_numpy()

    # Gaussian moving average over integer scenarios
    xs = np.arange(x.min(), x.max() + 1)
    ys = np.zeros_like(xs, dtype=float)
    for i, xi in enumerate(xs):
        w = np.exp(-0.5 * ((x - xi) / bandwidth) ** 2)
        wsum = w.sum()
        ys[i] = (w @ y) / wsum if wsum > 0 else np.nan

    plt.figure(figsize=(12, 5))
    plt.plot(x, y, marker="o", linewidth=0, alpha=0.5, label="Scenarios")
    plt.plot(xs, ys, linewidth=2, label=f"Gaussian smooth (bw={bandwidth})")
    plt.xlabel("Scenario")
    plt.ylabel("Objective value")
    plt.title("Objective per Scenario (Smoothed)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(out_svg_path), exist_ok=True)
    plt.savefig(out_svg_path, bbox_inches="tight")
    plt.show()

def plot_objective_by_rank(df_obj, out_svg_path=".cache/figs/objective_by_rank.svg"):
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
dap.model_manager.solve_model()


# %% ------------- collect scenario objectives (safe discovery) -------------
os.makedirs(".cache/figs", exist_ok=True)

df_obj = collect_objectives_via_discovery(dap)
print("Top 5 scenarios/models by objective:")
print(df_obj.sort_values("objective", ascending=False).head())

df_obj.to_csv(".cache/figs/objectives_by_scenario.csv", index=False)

# --------- PLOTS ----------
# 1) Density (value distribution)
plot_objective_density_from_df(df_obj, out_svg_path=".cache/figs/objective_density_by_scenario.svg")
# 2) Objective vs Scenario (x = scenario number)
plot_objective_vs_scenario(df_obj, out_svg_path=".cache/figs/objective_per_scenario.svg")
# 3) Smoothed objective vs scenario
plot_objective_vs_scenario_smoothed(df_obj, bandwidth=5.0, out_svg_path=".cache/figs/objective_per_scenario_smoothed.svg")
# 4) Rank plot (optional)
plot_objective_by_rank(df_obj, out_svg_path=".cache/figs/objective_by_rank.svg")


# %% ------------------ rest of your analysis/plots ------------------
# Consensus switch/tap states
print(dap.model_manager.zδ_variable)
print(dap.model_manager.zζ_variable)

# Compare with pandapower & plot annotated grid
node_data, edge_data = compare_dig_a_plan_with_pandapower(
    dig_a_plan=dap, net=net, from_z=True
)
fig = plot_grid_from_pandapower(net=net, dap=dap, from_z=True)

# Fixed-switch comparison
dap_fixed = copy.deepcopy(dap)
dap_fixed.solve_model(fixed_switches=True)

# Distribution plots
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
