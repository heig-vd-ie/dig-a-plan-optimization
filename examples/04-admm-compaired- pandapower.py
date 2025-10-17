# %% ------------------ setup & imports ------------------
import os
os.chdir(os.getcwd().replace("/src", ""))

from examples import *   

import copy as _copy
import re, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import patito as pt
import polars as pl
from polars import col as c

# Pyomo (only to fetch ADMM objective values)
try:
    import pyomo.environ as pe
except Exception:
    pe = None


# ================= helpers: objectives from ADMM (per scenario) ================
def _get_active_objective_safe(model):
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
            arg = int(np.argmax(gnp[:, 1])); tap = int(gnp[arg, 0])
            best[int(eid)] = tap  # type: ignore
        return best

    if has_edge:
        value_cols = [c for c in df.columns if c != "edge_id"]
        best = {}
        for row in df.to_dicts():
            eid = int(row["edge_id"])
            best_col, best_val = None, -1e18
            for c0 in value_cols:
                try:
                    fv = float(row[c0])
                except Exception:
                    continue
                if fv > best_val:
                    best_val = fv; best_col = c0
            if best_col is not None:
                m = re.search(r"(-?\d+)\s*$", best_col)
                tap = int(m.group(1)) if m else 0
                best[eid] = tap
        return best
    return {}

# ======= optional: group-consistency for δ =======
def harmonize_by_groups(z_map: dict[int, float], groups: dict[int, list[int]], rule: str = "majority") -> dict[int, float]:
    z = z_map.copy()
    for _, edges in groups.items():
        vals = [z[e] for e in edges if e in z]
        if not vals: continue
        if rule == "majority":
            ones = sum(1 for v in vals if v >= 0.5)
            zeros = len(vals) - ones
            rep = (1.0 if ones > zeros else 0.0) if ones != zeros else (1.0 if (sum(vals)/len(vals)) >= 0.5 else 0.0)
        elif rule == "mean":
            rep = float(sum(vals)/len(vals))
            rep = 0.0 if rep < 0.1 else (1.0 if rep > 0.9 else rep)
        else:
            raise ValueError("rule must be 'majority' or 'mean'")
        for e in edges:
            if e in z: z[e] = rep
    return z


# =============== PANDAPOWER helpers (apply y* on pp.NET) =======================
def _safe_int_from_mixed(x, field_name="value"):
    if isinstance(x, (int, np.integer)): return int(x)
    if isinstance(x, str):
        m = re.search(r'(-?\d+)\s*$', x.strip())
        if m: return int(m.group(1))
    raise ValueError(f"Cannot parse integer from {field_name}={x!r}")

def _build_edge_to_pp_switch_index(dap) -> dict[int, int]:
    dap.result_manager.init_model_instance(scenario=0)
    sw = dap.result_manager.extract_switch_status()
    for req in ("edge_id","eq_fk"):
        if req not in sw.columns:
            raise RuntimeError(f"extract_switch_status() missing column {req}")
    mapping = {}
    for r in sw.to_dicts():
        eid = _safe_int_from_mixed(r["edge_id"], "edge_id")
        fk  = _safe_int_from_mixed(r["eq_fk"],   "eq_fk")  # 'switch 7' -> 7
        mapping[eid] = fk
    return mapping

# Set according to δ definition:
# True  => δ=1 means CLOSED; δ=0 means OPEN  (common)
# False => δ=1 means OPEN;   δ=0 means CLOSED
POLARITY_DELTA_1_IS_CLOSED = True

def apply_y_to_pandapower_switches(net: pp.pandapowerNet, z_switch_map: dict[int, float], edge_to_pp_switch: dict[int, int]) -> None:
    if not hasattr(net, "switch") or net.switch is None or net.switch.empty: return
    valid_idx = set(net.switch.index.tolist())
    for edge_id, z in z_switch_map.items():
        pp_idx = edge_to_pp_switch.get(edge_id)
        if pp_idx is None or pp_idx not in valid_idx: continue
        z_round = int(round(float(z)))
        closed_val = bool(z_round) if POLARITY_DELTA_1_IS_CLOSED else bool(1 - z_round)
        net.switch.at[pp_idx, "closed"] = closed_val

def apply_scenario_injections_to_pp(net_base: pp.pandapowerNet, grid_data, scenario: int) -> pp.pandapowerNet:
    net_s = _copy.deepcopy(net_base)
    params_root = None
    for attr in ("grid_data_parameters_dict", "scenario_parameters", "scenarios", "scenarios_dict"):
        if hasattr(grid_data, attr):
            params_root = getattr(grid_data, attr); break
    if params_root is None: return net_s
    scen_key = scenario if (isinstance(params_root, dict) and scenario in params_root) else list(params_root.keys())[scenario]
    scen = params_root[scen_key]
    if hasattr(net_s, "load") and net_s.load is not None and not net_s.load.empty:
        for key in ("p_load", "load_p", "loads_p", "P_load"):
            if key in scen:
                arr = np.asarray(scen[key], dtype=float)
                if len(arr) == len(net_s.load): net_s.load["p_mw"] = arr
                break
        for key in ("q_load", "load_q", "loads_q", "Q_load"):
            if key in scen:
                arr = np.asarray(scen[key], dtype=float)
                if len(arr) == len(net_s.load): net_s.load["q_mvar"] = arr
                break
    if hasattr(net_s, "sgen") and net_s.sgen is not None and not net_s.sgen.empty:
        for key in ("p_sgen", "sgen_p", "gen_p", "P_sgen"):
            if key in scen:
                arr = np.asarray(scen[key], dtype=float)
                if len(arr) == len(net_s.sgen): net_s.sgen["p_mw"] = arr
                break
        for key in ("q_sgen", "sgen_q", "gen_q", "Q_sgen"):
            if key in scen:
                arr = np.asarray(scen[key], dtype=float)
                if len(arr) == len(net_s.sgen): net_s.sgen["q_mvar"] = arr
                break
    return net_s

def pandapower_objective(net: pp.pandapowerNet, vmin=0.95, vmax=1.05, v_penalty=1e3) -> float:
    losses = 0.0
    if hasattr(net, "res_line") and net.res_line is not None and not net.res_line.empty:
        losses += float(net.res_line["pl_mw"].sum())
    if hasattr(net, "res_trafo") and net.res_trafo is not None and not net.res_trafo.empty:
        losses += float(net.res_trafo["pl_mw"].sum())
    viol = 0.0
    if hasattr(net, "res_bus") and net.res_bus is not None and not net.res_bus.empty:
        vm = net.res_bus["vm_pu"].to_numpy(float)
        low = np.clip(vmin - vm, a_min=0.0, a_max=None)
        high = np.clip(vm - vmax, a_min=0.0, a_max=None)
        viol = float(low.sum() + high.sum())
    return losses + v_penalty * viol


# %% ------------------ Stage A: ADMM -> learn y* -------------------
net = pp.from_pickle("data/simple_grid.p")

groups = {
    0: [19, 20, 21, 29, 32, 35],
    1: [35, 30, 33, 25, 26, 27],
    2: [27, 32, 22, 23, 34],
    3: [31, 24, 28, 21, 22, 23],
    4: [34, 26, 25, 24, 31],
}

NUM_SCEN_TRAIN = 50

grid = pandapower_to_dig_a_plan_schema_with_scenarios(
    net, number_of_random_scenarios=NUM_SCEN_TRAIN,
    p_bounds=(-0.6, 1.5), q_bounds=(-0.1, 0.1),
    v_bounds=(-0.1, 0.1), v_min=0.95, v_max=1.05,
)

config = ADMMConfig(
    verbose=False, pipeline_type=PipelineType.ADMM, solver_name="gurobi",
    solver_non_convex=2, big_m=1e3, ε=1, ρ=2.0,
    γ_infeasibility=10, γ_admm_penalty=1.0, γ_trafo_loss=1e2,
    groups=groups, max_iters=20, μ=10.0, τ_incr=2.0, τ_decr=2.0,
)

dap = DigAPlanADMM(config=config)
dap.add_grid_data(grid)
dap.solve_model()

# consensus y* (from ADMM)
zδ_df = dap.model_manager.zδ_variable
zζ_df = dap.model_manager.zζ_variable
z_switch_y = harmonize_by_groups(_switch_df_to_dict(zδ_df), groups, rule="majority")
tap_choice_y = _taps_df_to_choice(zζ_df)
print("Sample y* (δ):", list(z_switch_y.items())[:5])
print("Sample y* (ζ tap choice):", list(tap_choice_y.items())[:5])

# ADMM full-run objectives (free binaries)
df_ADMM = collect_objectives_from_admm_instances(dap).rename(columns={"objective": "objective_ADMM"})

# ADMM Normal-Open baseline (switches fixed to original normal_open)
dap_NO = DigAPlanADMM(config=config)
dap_NO.add_grid_data(grid)
dap_NO.solve_model(fixed_switches=True)
df_NO_admm = collect_objectives_from_admm_instances(dap_NO).rename(columns={"objective": "objective_ADMM_NormalOpen"})


# %% ---- Apply y* in **pandapower** & evaluate ---------------------------------
edge_to_pp_switch = _build_edge_to_pp_switch_index(dap)

print(f"[DIAG] y* entries: {len(z_switch_y)}, mapping size: {len(edge_to_pp_switch)}, pp switches: {len(net.switch)}")
mapped = sum(1 for eid in z_switch_y if eid in edge_to_pp_switch)
print(f"[DIAG] y* entries mapped to PP: {mapped}")

net_baseline = _copy.deepcopy(net)
net_yfixed   = _copy.deepcopy(net)

apply_y_to_pandapower_switches(net_yfixed, z_switch_y, edge_to_pp_switch)

# apply_y_taps_to_pandapower(net_yfixed, tap_choice_y, edge_to_trafo_idx)

base_closed = net_baseline.switch["closed"].astype(bool).values
y_closed    = net_yfixed.switch["closed"].astype(bool).values
num_diff = int(np.sum(base_closed != y_closed))
print(f"[DIAG] y* changed {num_diff} PP switches (δ=1->closed: {POLARITY_DELTA_1_IS_CLOSED})")
for eid in list(z_switch_y.keys())[:5]:
    z = z_switch_y[eid]; pp_idx = edge_to_pp_switch.get(eid)
    if pp_idx is not None:
        print(f"[DIAG] edge {eid}: δ*={z:.2f}, baseline_closed={bool(net_baseline.switch.at[pp_idx,'closed'])}, y*_closed={bool(net_yfixed.switch.at[pp_idx,'closed'])}")

num_scen = len(getattr(dap.model_manager, "Ω", [])) or NUM_SCEN_TRAIN
rows_fixed = []
for s in range(num_scen):
    net_s_fixed = apply_scenario_injections_to_pp(net_yfixed, grid, scenario=s)
    try:
        pp.runpp(net_s_fixed, algorithm="nr")
        obj = pandapower_objective(net_s_fixed, vmin=0.95, vmax=1.05, v_penalty=1e3)
    except Exception:
        obj = float("inf")
    rows_fixed.append({"scenario": s, "objective_y_star_fixed_pp": obj})

df_pp_fixed_y = pd.DataFrame(rows_fixed)
os.makedirs(".cache/figs", exist_ok=True)
df_pp_fixed_y.to_csv(".cache/figs/pandapower_objectives_fixedY.csv", index=False)


# %% ---------------- Comparison: ADMM vs PP:Fixed y* vs ADMM:Normal-Open -------
df_cmp_all = (
    df_ADMM[["scenario", "objective_ADMM"]]
    .merge(df_pp_fixed_y, on="scenario", how="inner")
    .merge(df_NO_admm, on="scenario", how="inner")
    .sort_values("scenario")
    .reset_index(drop=True)
)
df_cmp_all.to_csv(".cache/figs/cmp_ADMM_vs_PP_FixedY_vs_ADMM_NormalOpen.csv", index=False)

plt.figure(figsize=(9, 5))
plt.boxplot(
    [
        df_cmp_all["objective_ADMM"].to_numpy(float),               # ADMM free
        df_cmp_all["objective_y_star_fixed_pp"].to_numpy(float),    # PP Fixed y*
        df_cmp_all["objective_ADMM_NormalOpen"].to_numpy(float),    # ADMM Normal-Open
    ],
    labels=["ADMM (free)", "PP: Fixed y*", "ADMM: Normal-Open"],    # type: ignore
    patch_artist=True, showmeans=True
)
plt.yscale("symlog", linthresh=0.5)
plt.grid(True, axis="y", alpha=0.3)
plt.ylabel("Objective value (framework-specific)")
plt.title(f"ADMM (free) vs PP: Fixed y* vs ADMM: Normal-Open ({num_scen} scenarios)")
plt.tight_layout()
plt.savefig(".cache/figs/boxplot_ADMM_vs_PP_FixedY_vs_ADMM_NormalOpen.svg", bbox_inches="tight")
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(df_cmp_all["scenario"], df_cmp_all["objective_ADMM"],               marker="o", linewidth=1, label="ADMM (free)")
plt.plot(df_cmp_all["scenario"], df_cmp_all["objective_y_star_fixed_pp"],   marker="s", linewidth=1, label="PP: Fixed y*")
plt.plot(df_cmp_all["scenario"], df_cmp_all["objective_ADMM_NormalOpen"],   marker="x", linewidth=1, label="ADMM: Normal-Open")
plt.xlabel("Scenario"); plt.ylabel("Objective value")
plt.title("Per-scenario comparison")
plt.grid(True, alpha=0.3); plt.legend()
plt.tight_layout()
plt.savefig(".cache/figs/overlay_ADMM_vs_PP_FixedY_vs_ADMM_NormalOpen.svg", bbox_inches="tight")
plt.show()
