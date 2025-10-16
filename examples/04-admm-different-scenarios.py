# %% ------------------ setup & imports ------------------
import os
os.chdir(os.getcwd().replace("/src", ""))

from examples import *   

import copy
import re, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# data frame tooling
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
    cols = [c.lower() for c in df.columns]
    has_edge = "edge_id" in df.columns
    tap_col = None
    val_col = None

    # tidy case
    for c0 in df.columns:
        cl = c0.lower()
        if cl in ("tap", "tap_idx", "tap_index"): tap_col = c0
        if cl in ("ζ", "zeta", "value", "weight", "prob", "consensus"): val_col = c0
    if has_edge and (tap_col is not None) and (val_col is not None):
        # argmax per transformer
        best = {}
        for eid, g in df.group_by("edge_id"):
            gnp = g.select([tap_col, val_col]).to_numpy()
            if gnp.size == 0: continue
            arg = int(np.argmax(gnp[:, 1]))
            tap = int(gnp[arg, 0])
            best[int(eid)] = tap #type: ignore
        return best

    # wide case: columns like 'tap_-2','tap_-1','tap0','tap1', etc.
    if has_edge:
        value_cols = [c for c in df.columns if c != "edge_id"]
        best = {}
        for row in df.to_dicts():
            eid = int(row["edge_id"])
            best_col = None
            best_val = -1e18
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
                # extract last int in the column name
                m = re.search(r"(-?\d+)\s*$", best_col)
                tap = int(m.group(1)) if m else 0
                best[eid] = tap
        return best

    # fallback: cannot read
    return {}


# ======= optional: group-consistency for switching (if your model groups δ) =======
def harmonize_by_groups(z_map: dict[int, float],
                        groups: dict[int, list[int]],
                        rule: str = "majority") -> dict[int, float]:
    """
    Make the warm-start / fixed design group-consistent.
    """
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

    # remember the original dtype so we can cast back (often pl.Boolean)
    orig_dtype = ed.schema.get("normal_open", pl.Boolean)

    # build map: edge_id -> desired normal_open (True=open, False=closed)
    # here z in {0,1} with 1=closed, 0=open  ==>  normal_open = 1 - z
    new_no_map_int = {eid: int(round(1.0 - val)) for eid, val in z_switch_dict.items()}

    # create a new 'normal_open' respecting switches; fall back to existing value for non-switch rows or missing ids
    ed2 = ed.with_columns(
        pl.when(c("type") == "switch")
          .then(
              c("edge_id")
              .replace_strict(new_no_map_int, default=c("normal_open"))
          )
          .otherwise(c("normal_open"))
          .alias("normal_open")
    )

    # cast to original dtype expected by Patito schema (Boolean in most setups)
    # if original is Boolean, convert 0/1 -> False/True
    ed2 = ed2.with_columns(
        c("normal_open").cast(orig_dtype, strict=False)
    )

    # rebuild Patito DataFrame with the SAME model class as before
    model_cls = grid_data.edge_data.__class__.model  # the actual Patito model class (e.g., EdgeData)
    grid_data.edge_data = pt.DataFrame(ed2).set_model(model_cls)

    # validate against schema
    grid_data.edge_data.validate()

    return grid_data




def try_fix_taps_on_models(dap, tap_choice: dict[int, int]):
    """
    Try to fix transformer tap variables ζ to chosen tap per edge.
    If the pipeline exposes a knob (e.g., solve_model(fixed_taps=True, tap_map=...)),
    use it. Otherwise, attempt to set bounds on ζ on constructed instances.
    This is best-effort: if unsupported, it silently returns.
    """
    mm = getattr(dap, "model_manager", None)
    if mm is None: return
    # If the API has a direct method, prefer it:
    if hasattr(mm, "set_fixed_taps"):
        try:
            mm.set_fixed_taps(tap_choice)
            return
        except Exception:
            pass
    # Otherwise, if admm_model_instances exist and ζ is one-hot:
    try:
        instances = mm.admm_model_instances
        for _, mdl in instances.items():
            if not hasattr(mdl, "ζ"):
                continue
            # Expect mdl.ζ[(edge_id, tap)] \in {0,1}, sum_tap ζ = 1
            for (eid, tap), val in list(mdl.ζ.extract_values().items()):
                # freeze to 1 only for the chosen tap; 0 otherwise
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


# %% ------------------ Stage A: ADMM on 10 scenarios -> y* ------------------
net = pp.from_pickle("data/simple_grid.p")

groups = {
    0: [19, 20, 21, 29, 32, 35],
    1: [35, 30, 33, 25, 26, 27],
    2: [27, 32, 22, 23, 34],
    3: [31, 24, 28, 21, 22, 23],
    4: [34, 26, 25, 24, 31],
}

grid10 = pandapower_to_dig_a_plan_schema_with_scenarios(
    net, number_of_random_scenarios=10,
    p_bounds=(-0.6, 1.5), q_bounds=(-0.1, 0.1),
    v_bounds=(-0.1, 0.1), v_min=0.95, v_max=1.05,
)

config10 = ADMMConfig(
    verbose=False, pipeline_type=PipelineType.ADMM, solver_name="gurobi",
    solver_non_convex=2, big_m=1e3, ε=1, ρ=2.0,
    γ_infeasibility=10, γ_admm_penalty=1.0, γ_trafo_loss=1e2,
    groups=groups, max_iters=20, μ=10.0, τ_incr=2.0, τ_decr=2.0,
)

dap10 = DigAPlanADMM(config=config10)
dap10.add_grid_data(grid10)
dap10.solve_model()

# ---- consensus y* from 10-scenario run
zδ_df = dap10.model_manager.zδ_variable           # switches
zζ_df = dap10.model_manager.zζ_variable           # taps (consensus)

z_switch_y = _switch_df_to_dict(zδ_df)
# if you use grouped switching, harmonize inside each group (optional but recommended)
z_switch_y = harmonize_by_groups(z_switch_y, groups, rule="majority")

tap_choice_y = _taps_df_to_choice(zζ_df)          # {transformer_edge_id: chosen_tap}
print("Sample y* (δ):", list(z_switch_y.items())[:5])
print("Sample y* (ζ tap choice):", list(tap_choice_y.items())[:5])


# %% ---- Evaluate/scale on 100 scenarios with binaries fixed to y* -------------
grid100 = pandapower_to_dig_a_plan_schema_with_scenarios(
    net, number_of_random_scenarios=50,
    p_bounds=(-0.6, 1.5), q_bounds=(-0.1, 0.1),
    v_bounds=(-0.1, 0.1), v_min=0.95, v_max=1.05,
)

config100 = ADMMConfig(
    verbose=False, pipeline_type=PipelineType.ADMM, solver_name="gurobi",
    solver_non_convex=2, big_m=1e3, ε=1, ρ=2.0,
    γ_infeasibility=10, γ_admm_penalty=1.0, γ_trafo_loss=1e2,
    groups=groups, max_iters=20, μ=10.0, τ_incr=2.0, τ_decr=2.0,
)

# --- (A) FIXED DESIGN: δ=δ*, ζ=ζ* (continuous OPF per scenario)
# apply switches into normal_open
grid100_fixed = apply_design_to_normal_open(grid100, z_switch_y)

dap100_fixed = DigAPlanADMM(config=config100)
dap100_fixed.add_grid_data(grid100_fixed)
# try to freeze taps as well 
try_fix_taps_on_models(dap100_fixed, tap_choice_y)
# switches fixed via normal_open
dap100_fixed.solve_model(fixed_switches=True)

df_fixed = collect_objectives_from_admm_instances(dap100_fixed).rename(
    columns={"objective": "objective_y_star_fixed"}
)

# --- (B) BASELINE: Normal-Open (original topology), fixed switches
dap100_NO = DigAPlanADMM(config=config100)
dap100_NO.add_grid_data(grid100)  # original normal_open
dap100_NO.solve_model(fixed_switches=True)
df_NO = collect_objectives_from_admm_instances(dap100_NO).rename(
    columns={"objective": "objective_NormalOpen"}
)

# --- (C) ADMM(10) objectives, just to see distribution
df_10 = collect_objectives_from_admm_instances(dap10).rename(
    columns={"objective": "objective_ADMM_10"}
)

# Merge for plotting (align on scenario where applicable)
# Note: df_10 has 10 scenarios; df_fixed and df_NO have 100 scenarios.
df_plot = df_fixed.merge(df_NO, on="scenario", how="inner")
# Save tables
os.makedirs(".cache/figs", exist_ok=True)
df_plot.to_csv(".cache/figs/objectives_y_star_fixed_vs_NormalOpen.csv", index=False)
df_10.to_csv(".cache/figs/objectives_ADMM_10.csv", index=False)

# %% ------------------ Plots: boxplot comparison -------------------------------
# Compare distributions:
#   - ADMM(10)          : 'objective_ADMM_10'  (10 values)
#   - y* fixed on 100   : 'objective_y_star_fixed' (100 values)
#   - Normal-Open (100) : 'objective_NormalOpen'   (100 values)

series_list = []
labels = []

if "objective_ADMM_10" in df_10:
    series_list.append(df_10["objective_ADMM_10"].to_numpy(float))
    labels.append("ADMM (10 scen.)")

series_list.append(df_plot["objective_y_star_fixed"].to_numpy(float))
labels.append("Fixed y* (100 scen.)")

series_list.append(df_plot["objective_NormalOpen"].to_numpy(float))
labels.append("Normal Open (100)")

plt.figure(figsize=(9, 5))
bp = plt.boxplot(series_list, labels=labels, patch_artist=True, showmeans=True) #type:ignore
for box in bp["boxes"]:   box.set_alpha(0.5)
for median in bp["medians"]: median.set_linewidth(2.0)
for mean in bp["means"]:  mean.set_marker("x"); mean.set_markersize(8)

# If the scales differ a lot, symlog avoids squashing
plt.yscale("symlog", linthresh=0.5)
plt.ylabel("Objective value")
plt.title("Objective distributions: ADMM(10) vs Fixed y* (100) vs Normal Open (100)")
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(".cache/figs/boxplot_admm10_fixedY_normalopen.svg", bbox_inches="tight")
plt.show()

# (Optional) also show per-scenario overlay for the 100-scenario evaluation
plt.figure(figsize=(12, 5))
plt.plot(df_plot["scenario"], df_plot["objective_y_star_fixed"], marker="o", linewidth=1, label="Fixed y* (100)")
plt.plot(df_plot["scenario"], df_plot["objective_NormalOpen"], marker="x", linewidth=1, label="Normal Open (100)")
plt.xlabel("Scenario"); plt.ylabel("Objective")
plt.title("Per-scenario objective: Fixed y* vs Normal Open (100 scenarios)")
plt.grid(True, alpha=0.3); plt.legend()
plt.tight_layout()
plt.savefig(".cache/figs/objective_per_scenario_fixedY_vs_normalopen.svg", bbox_inches="tight")
plt.show()
