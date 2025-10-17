# %% ------------------ setup & imports ------------------
import os
os.chdir(os.getcwd().replace("/src", ""))

from examples import *   

import copy
import re, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# data-frame tooling
import patito as pt
import polars as pl
from polars import col as c

# Pyomo for objective evaluation
try:
    import pyomo.environ as pe
except Exception:
    pe = None


# =============== helpers: objectives (one per scenario) ===============
def _get_active_objective_safe(model):
    """Return a constructed Objective on a concrete model, else None."""
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
    """Use ADMM's per-scenario model instances to get one objective per scenario."""
    mm = getattr(dap, "model_manager", None)
    if mm is None or not hasattr(mm, "admm_model_instances"):
        raise RuntimeError("model_manager.admm_model_instances not found.")
    rows = []
    for omega, model in mm.admm_model_instances.items():
        obj = _get_active_objective_safe(model)
        if obj is None:
            continue
        rows.append({"scenario": int(omega), "objective": float(pe.value(obj))})  # type: ignore
    if not rows:
        raise RuntimeError("No constructed Objectives found in admm_model_instances.")
    return pd.DataFrame(rows).sort_values("scenario").reset_index(drop=True)


# =============== helpers: parse consensus (δ, ζ) -> y* =================
def _switch_df_to_dict(df: pl.DataFrame) -> dict[int, float]:
    """
    Convert zδ_variable (Polars DF) to {edge_id: float in [0,1]}.
    Handles numeric or strings like 'switch 0', 'closed', 'open'.
    """
    if "edge_id" not in df.columns:
        key_col = next((col for col in df.columns if "edge" in col.lower() and "id" in col.lower()), None)
        if key_col is None:
            raise RuntimeError(f"Cannot find edge_id column in zδ_variable; columns={df.columns}")
        df = df.rename({key_col: "edge_id"})

    candidate_cols = [col for col in df.columns if col != "edge_id"]
    numeric_col = None
    for col in candidate_cols:
        try:
            casted = df[col].cast(pl.Float64, strict=False)
            if casted.null_count() <= max(1, int(0.1 * len(df))):
                numeric_col = col
                break
        except Exception:
            continue

    if numeric_col is not None:
        vals = df.select(["edge_id", pl.col(numeric_col).cast(pl.Float64)]).to_dicts()
        mapping = {int(r["edge_id"]): float(r[numeric_col]) for r in vals}
    else:
        def parse_any(v):
            if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
                return float(v)
            if isinstance(v, str):
                s = v.strip().lower()
                if s in ("open", "false", "0"):   return 0.0
                if s in ("closed", "true", "1"):  return 1.0
                m = re.search(r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$', s)
                if m:
                    try: return float(m.group(1))
                    except Exception: pass
            return None

        mapping = {}
        for row in df.to_dicts():
            edge = int(row["edge_id"])
            val = None
            for k, v in row.items():
                if k == "edge_id": continue
                parsed = parse_any(v)
                if parsed is not None:
                    val = parsed
                    break
            if val is None:
                raise RuntimeError(f"Cannot parse switch value for edge_id={edge} from row={row}")
            mapping[edge] = float(val)

    # clip & snap to {0,1}
    for k, v in list(mapping.items()):
        vv = max(0.0, min(1.0, v))
        mapping[k] = 0.0 if vv < 0.1 else (1.0 if vv > 0.9 else vv)
    return mapping

def _taps_df_to_choice(df: pl.DataFrame) -> dict[int, int]:
    """
    Convert zζ_variable consensus to {transformer_edge_id: chosen_tap_index}.
    Handles tidy formats like ['edge_id','TAP','ζ'] or wide one-hot.
    """
    has_edge = "edge_id" in df.columns
    tap_col, val_col = None, None

    for c0 in df.columns:
        cl = c0.lower()
        if cl in ("tap", "tap_idx", "tap_index"): tap_col = c0
        if cl in ("ζ", "zeta", "value", "weight", "prob", "consensus"): val_col = c0
    if has_edge and (tap_col is not None) and (val_col is not None):
        best = {}
        for eid, g in df.group_by("edge_id"):
            gnp = g.select([tap_col, val_col]).to_numpy()
            if gnp.size == 0: continue
            arg = int(np.argmax(gnp[:, 1]))
            tap = int(gnp[arg, 0])
            best[int(eid)] = tap  # type: ignore
        return best

    if has_edge:
        value_cols = [c for c in df.columns if c != "edge_id"]
        best = {}
        for row in df.to_dicts():
            eid = int(row["edge_id"])
            best_col, best_val = None, -1e18
            for c0 in value_cols:
                v = row[c0]
                try:
                    fv = float(v)
                except Exception:
                    continue
                if fv > best_val:
                    best_val = fv
                    best_col = c0
            if best_col is not None:
                m = re.search(r"(-?\d+)\s*$", best_col)
                tap = int(m.group(1)) if m else 0
                best[eid] = tap
        return best

    return {}


# ======= optional: group-consistency for switching (if the model groups δ) =======
def harmonize_by_groups(z_map: dict[int, float],
                        groups: dict[int, list[int]],
                        rule: str = "majority") -> dict[int, float]:
    """Make the warm-start / fixed design group-consistent."""
    z = z_map.copy()
    for _, edges in groups.items():
        vals = [z[e] for e in edges if e in z]
        if not vals:
            continue
        if rule == "majority":
            ones = sum(1 for v in vals if v >= 0.5)
            zeros = len(vals) - ones
            if ones == zeros:
                rep = 1.0 if (sum(vals)/len(vals)) >= 0.5 else 0.0
            else:
                rep = 1.0 if ones > zeros else 0.0
        elif rule == "mean":
            rep = float(sum(vals)/len(vals))
            rep = 0.0 if rep < 0.1 else (1.0 if rep > 0.9 else rep)
        else:
            raise ValueError("rule must be 'majority' or 'mean'")
        for e in edges:
            if e in z:
                z[e] = rep
    return z


# =============== utilities for fixed-design evaluation =================
def apply_design_to_normal_open(grid_data, z_switch_dict: dict[int, float]):
    """
    Update 'normal_open' for switches to match y* and KEEP the original dtype.
      δ* ~ 1 (closed) -> normal_open = False
      δ* ~ 0 (open)   -> normal_open = True
    """
    ed = grid_data.edge_data.as_polars()
    orig_dtype = ed.schema.get("normal_open", pl.Boolean)
    new_no_map_int = {eid: int(round(1.0 - val)) for eid, val in z_switch_dict.items()}
    ed2 = ed.with_columns(
        pl.when(c("type") == "switch")
          .then(c("edge_id").replace_strict(new_no_map_int, default=c("normal_open")))
          .otherwise(c("normal_open"))
          .alias("normal_open")
    )
    ed2 = ed2.with_columns(c("normal_open").cast(orig_dtype, strict=False))
    model_cls = grid_data.edge_data.__class__.model
    grid_data.edge_data = pt.DataFrame(ed2).set_model(model_cls)
    grid_data.edge_data.validate()
    return grid_data

def try_fix_taps_on_models(dap, tap_choice: dict[int, int]):
    """
    Best-effort freezing of transformer taps ζ to chosen tap per transformer.
    Safe no-op if unsupported by the pipeline/version.
    """
    mm = getattr(dap, "model_manager", None)
    if mm is None:
        return
    if hasattr(mm, "set_fixed_taps"):
        try:
            mm.set_fixed_taps(tap_choice)
            return
        except Exception:
            pass
    try:
        instances = mm.admm_model_instances
        for _, mdl in instances.items():
            if not hasattr(mdl, "ζ"):
                continue
            for (eid, tap), _ in list(mdl.ζ.extract_values().items()):
                chosen = tap_choice.get(int(eid), None)
                if chosen is None:
                    continue
                var = mdl.ζ[eid, tap]
                if int(tap) == int(chosen):
                    var.setlb(1.0); var.setub(1.0)
                else:
                    var.setlb(0.0); var.setub(0.0)
    except Exception:
        pass


# %% ------------------ Stage A: ADMM on 100 scenarios -> y* ------------------
net = pp.from_pickle("data/simple_grid.p")

groups = {
    0: [19, 20, 21, 29, 32, 35],
    1: [35, 30, 33, 25, 26, 27],
    2: [27, 32, 22, 23, 34],
    3: [31, 24, 28, 21, 22, 23],
    4: [34, 26, 25, 24, 31],
}

grid100 = pandapower_to_dig_a_plan_schema_with_scenarios(
    net, number_of_random_scenarios=100,       # <-- 100 for design stage
    p_bounds=(-0.6, 1.5), q_bounds=(-0.1, 0.1),
    v_bounds=(-0.1, 0.1), v_min=0.95, v_max=1.05,
)

config100 = ADMMConfig(
    verbose=False, pipeline_type=PipelineType.ADMM, solver_name="gurobi",
    solver_non_convex=2, big_m=1e3, ε=1, ρ=2.0,
    γ_infeasibility=10, γ_admm_penalty=1.0, γ_trafo_loss=1e2,
    groups=groups, max_iters=20, μ=10.0, τ_incr=2.0, τ_decr=2.0,
)

dap100 = DigAPlanADMM(config=config100)
dap100.add_grid_data(grid100)
dap100.solve_model()

# design y* from 100-scenario run
zδ_df = dap100.model_manager.zδ_variable     # switches
zζ_df = dap100.model_manager.zζ_variable     # taps (consensus)
z_switch_y = harmonize_by_groups(_switch_df_to_dict(zδ_df), groups, rule="majority")
tap_choice_y = _taps_df_to_choice(zζ_df)

print("Sample y* (δ):", list(z_switch_y.items())[:5])
print("Sample y* (ζ tap choice):", list(tap_choice_y.items())[:5])


# %% ---- Evaluate/scale on 100 scenarios --------------------------------
grid100 = pandapower_to_dig_a_plan_schema_with_scenarios(
    net, number_of_random_scenarios=100,
    p_bounds=(-0.6, 1.5), q_bounds=(-0.1, 0.1),
    v_bounds=(-0.1, 0.1), v_min=0.95, v_max=1.05,
)

config100 = ADMMConfig(
    verbose=False, pipeline_type=PipelineType.ADMM, solver_name="gurobi",
    solver_non_convex=2, big_m=1e3, ε=1, ρ=2.0,
    γ_infeasibility=10, γ_admm_penalty=1.0, γ_trafo_loss=1e2,
    groups=groups, max_iters=20, μ=10.0, τ_incr=2.0, τ_decr=2.0,
)

# (A) FIXED DESIGN y*: δ=δ*, ζ=ζ* (solve each scenario continuous OPF)
grid100_fixed = apply_design_to_normal_open(grid100, z_switch_y)
dap100_fixed = DigAPlanADMM(config=config100)
dap100_fixed.add_grid_data(grid100_fixed)
try_fix_taps_on_models(dap100_fixed, tap_choice_y)  # optional best-effort
dap100_fixed.solve_model(fixed_switches=True)
df_fixed = collect_objectives_from_admm_instances(dap100_fixed).rename(
    columns={"objective": "objective_y_star_fixed"}
)

# (B) Normal-Open baseline (original topology), fixed switches
dap100_NO = DigAPlanADMM(config=config100)
dap100_NO.add_grid_data(grid100)
dap100_NO.solve_model(fixed_switches=True)
df_NO = collect_objectives_from_admm_instances(dap100_NO).rename(
    columns={"objective": "objective_NormalOpen"}
)

# (C) Full ADMM(100) (free binaries) — for comparison & switch status
dap100_full = DigAPlanADMM(config=config100)
dap100_full.add_grid_data(grid100)
dap100_full.solve_model()
df_ADMM100 = collect_objectives_from_admm_instances(dap100_full).rename(
    columns={"objective": "objective_ADMM_100"}
)

# (D) ADMM(10) objectives — just to show its distribution (10 scenarios)
df_100 = collect_objectives_from_admm_instances(dap100).rename(
    columns={"objective": "objective_ADMM_10"}
)

# Align & save objective tables
os.makedirs(".cache/figs", exist_ok=True)
df_plot = (df_ADMM100.merge(df_fixed, on="scenario")
                     .merge(df_NO, on="scenario")
                     .sort_values("scenario").reset_index(drop=True))
df_plot.to_csv(".cache/figs/objectives_100_ADMM_vs_FixedY_vs_NormalOpen.csv", index=False)
df_100.to_csv(".cache/figs/objectives_ADMM_10.csv", index=False)

# %% ------------------ Plots: boxplot & overlay -------------------------------
plt.figure(figsize=(9, 5))
plt.boxplot(
    [
        df_ADMM100["objective_ADMM_100"].to_numpy(float),
        df_plot["objective_y_star_fixed"].to_numpy(float),
        df_plot["objective_NormalOpen"].to_numpy(float),
    ],
    labels=["ADMM(100)", "Fixed y* (100)", "Normal Open (100)"], #type:ignore
    patch_artist=True, showmeans=True
)
plt.yscale("symlog", linthresh=0.5)
plt.ylabel("Objective value")
plt.title("Objective distributions (100 scenarios)")
plt.grid(True, which="both", axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(".cache/figs/boxplot_100_ADMM_vs_FixedY_vs_NormalOpen.svg", bbox_inches="tight")
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(df_plot["scenario"], df_plot["objective_ADMM_100"],     marker="o", linewidth=1, label="ADMM(100)")
plt.plot(df_plot["scenario"], df_plot["objective_y_star_fixed"], marker="s", linewidth=1, label="Fixed y* (100)")
plt.plot(df_plot["scenario"], df_plot["objective_NormalOpen"],   marker="x", linewidth=1, label="Normal Open (100)")
plt.xlabel("Scenario"); plt.ylabel("Objective")
plt.title("Per-scenario objective on 100 scenarios")
plt.grid(True, alpha=0.3); plt.legend()
plt.tight_layout()
plt.savefig(".cache/figs/objective_per_scenario_100_all.svg", bbox_inches="tight")
plt.show()


# %% ------------------ Switch status tables (per method) -----------------------
def _extract_switch_table(dap, scenarios=None, method_label="ADMM(100)", save_dir=None):
    """
    Returns a Polars DataFrame with columns:
    ['method','scenario','eq_fk','edge_id','δ','normal_open','open']
    (extract_switch_status already returns these.)
    """
    Ω = list(getattr(dap.model_manager, "Ω", []))
    if scenarios is None:
        scenarios = list(range(len(Ω))) if Ω else [0]

    out = []
    for s in scenarios:
        # IMPORTANT: select the scenario on the result manager first
        dap.result_manager.init_model_instance(scenario=s)
        # then call without a scenario kwarg
        sw = dap.result_manager.extract_switch_status()

        sw = sw.with_columns(
            pl.lit(method_label).alias("method"),
            pl.lit(int(s)).alias("scenario"),
        )
        out.append(sw)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            sw.write_csv(os.path.join(
                save_dir, f"switch_status_{method_label.replace(' ','_')}_s{s}.csv"
            ))

    return pl.concat(out, how="diagonal_relaxed") if out else pl.DataFrame({})

# Build per-method switch tables
# NOTE: make sure dap100_full, dap100_fixed, dap100_NO have been created & solved earlier.
sw_admm   = _extract_switch_table(dap100_full,  method_label="ADMM(100)",   save_dir=".cache/switch_tables/ADMM")
sw_fixed  = _extract_switch_table(dap100_fixed, method_label="Fixed y*",     save_dir=".cache/switch_tables/FixedY")
sw_normal = _extract_switch_table(dap100_NO,    method_label="Normal Open",  save_dir=".cache/switch_tables/NormalOpen")

sw_all = pl.concat([sw_admm, sw_fixed, sw_normal], how="diagonal_relaxed")
os.makedirs(".cache/figs", exist_ok=True)
sw_all.write_csv(".cache/figs/switch_status_all_methods.csv")
print(sw_all.head())
print(f"Total switch rows: {sw_all.height}")

# quick look: scenario 0 differences (δ by method)
s0 = 0
cmp_s0 = (
    sw_all.filter(c("scenario") == s0)
          .select(["method","edge_id","δ","open","normal_open"]) # type: ignore
          .pivot(index="edge_id", columns="method", values="δ") # type: ignore
          .sort("edge_id") # type: ignore
) # type: ignore
print("Scenario 0: δ by method (columns = method):")
print(cmp_s0.head())
cmp_s0.write_csv(".cache/figs/switch_compare_delta_s0.csv")

# also open/closed difference for scenario 0 (Boolean)
cmp_s0_open = (
    sw_all.filter(c("scenario") == s0) # type: ignore
          .select(["method","edge_id","open"]) # type: ignore
          .pivot(index="edge_id", columns="method", values="open") # type: ignore
          .sort("edge_id")
)
cmp_s0_open.write_csv(".cache/figs/switch_compare_open_s0.csv")

# summary: share of closed switches per scenario & method (δ>0.5)
summary = (
    sw_all
    .with_columns((c("δ") > 0.5).alias("closed"))
    .group_by(["method","scenario"])
    .agg([
        c("edge_id").count().alias("num_switches"),
        c("closed").sum().alias("num_closed"),
    ])
    .with_columns((c("num_closed") / c("num_switches")).alias("share_closed"))
    .sort(["method","scenario"])
)
summary.write_csv(".cache/figs/switch_closed_share_by_scenario.csv")

# disagreements (example): ADMM vs Fixed y*
methods = ["ADMM(100)", "Fixed y*", "Normal Open"]
sw_A = sw_all.filter(c("method")==methods[0]).rename({"δ":"δ_A"}).select(["scenario","edge_id","δ_A"])
sw_B = sw_all.filter(c("method")==methods[1]).rename({"δ":"δ_B"}).select(["scenario","edge_id","δ_B"])
disagree = (
    sw_A.join(sw_B, on=["scenario","edge_id"], how="inner")
        .with_columns(((c("δ_A") > 0.5) != (c("δ_B") > 0.5)).alias("disagree"))
        .filter(c("disagree"))
        .sort(["scenario","edge_id"])
)
disagree.write_csv(".cache/figs/switch_disagreements_ADMM_vs_FixedY.csv")

# %%
