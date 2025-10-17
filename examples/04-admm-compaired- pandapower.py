# %% ------------------ setup & imports ------------------
import os, json, copy as _copy, re, math
os.chdir(os.getcwd().replace("/src", ""))

from examples import *   

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import polars as pl
from polars import col as c

# -------------------- configuration --------------------
SAVE_DIR = ".cache/admm_insample_100"
os.makedirs(SAVE_DIR, exist_ok=True)

NUM_SCEN_TRAIN = 100    # train scenarios for ADMM (to learn y*)
NUM_SCEN_TEST  = 100    # new scenarios for OOS evaluation
S_BASE_W       = 1e6    # s_base used in scenario generator

# δ polarity:
# True  => δ=1 means CLOSED; δ=0 means OPEN
# False => δ=1 means OPEN;   δ=0 means CLOSED
POLARITY_DELTA_1_IS_CLOSED = True

# -------------------- file paths -----------------------
fp_switch = os.path.join(SAVE_DIR, "consensus_switch.json")
fp_taps   = os.path.join(SAVE_DIR, "consensus_taps.json")
fp_map    = os.path.join(SAVE_DIR, "edge_to_pp_switch.json")

# ================= helpers: parse consensus (δ, ζ) =================
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
    if "edge_id" not in df.columns:
        return {}
    tap_col, val_col = None, None
    for c0 in df.columns:
        cl = c0.lower()
        if cl in ("tap", "tap_idx", "tap_index"): tap_col = c0
        if cl in ("ζ", "zeta", "value", "weight", "prob", "consensus"): val_col = c0
    if tap_col is not None and val_col is not None:
        best = {}
        for eid, g in df.group_by("edge_id"):
            gnp = g.select([tap_col, val_col]).to_numpy()
            if gnp.size == 0: continue
            arg = int(np.argmax(gnp[:, 1])); tap = int(gnp[arg, 0])
            best[int(eid)] = tap #type: ignore
        return best
    # wide format fallback
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

# =============== PANDAPOWER helpers (edge_id mapping, injections) ==============
def _safe_int_from_mixed(x, field_name="value"):
    if isinstance(x, (int, np.integer)): return int(x)
    if isinstance(x, str):
        m = re.search(r'(-?\d+)\s*$', x.strip())
        if m: return int(m.group(1))
    raise ValueError(f"Cannot parse integer from {field_name}={x!r}")

def _build_edge_to_pp_switch_index(dap) -> dict[int, int]:
    """edge_id -> net.switch index (int)."""
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

def annotate_pp_switch_with_edge_ids(net: pp.pandapowerNet, edge_to_pp_switch: dict[int, int]) -> None:
    """Add/refresh net.switch['edge_id'] (int) from edge_id -> pp.switch-index mapping."""
    if not hasattr(net, "switch") or net.switch is None or net.switch.empty:
        return
    if "edge_id" not in net.switch.columns:
        net.switch["edge_id"] = -1
    for eid, pp_idx in edge_to_pp_switch.items():
        if pp_idx in net.switch.index:
            net.switch.at[pp_idx, "edge_id"] = int(eid)
    try:
        net.switch["edge_id"] = net.switch["edge_id"].astype(int)
    except Exception:
        pass

def apply_y_to_pandapower_switches(net: pp.pandapowerNet, z_switch_map: dict[int, float]) -> int:
    """Write net.switch.closed by matching on net.switch['edge_id']. Returns #changed."""
    if not hasattr(net, "switch") or net.switch is None or net.switch.empty:
        return 0
    if "edge_id" not in net.switch.columns:
        raise RuntimeError("net.switch has no 'edge_id'. Call annotate_pp_switch_with_edge_ids(...) first.")
    changed = 0
    for eid, z in z_switch_map.items():
        mask = (net.switch["edge_id"] == int(eid))
        if not mask.any():
            continue
        z_round = int(round(float(z)))
        desired = bool(z_round) if POLARITY_DELTA_1_IS_CLOSED else bool(1 - z_round)
        before = net.switch.loc[mask, "closed"].astype(bool)
        net.switch.loc[mask, "closed"] = desired
        changed += int((before.values != desired).sum())
    return changed

def _scenario_df_from_nodeedge(grid_nodeedge, scenario: int) -> pl.DataFrame | None:
    """
    Pull scenario 'scenario' from NodeEdgeModel.load_data-like container on grid_nodeedge.
    Expect fields: node_id, p_cons_pu, q_cons_pu, p_prod_pu, q_prod_pu.
    """
    for attr in ("load_data", "scenario_parameters", "scenarios", "grid_data_parameters_dict", "scenarios_dict"):
        if hasattr(grid_nodeedge, attr):
            root = getattr(grid_nodeedge, attr)
            if root is None:
                continue
            if isinstance(root, dict):
                key = scenario if scenario in root else (list(root.keys())[scenario] if len(root) > scenario else None)
                if key is None:
                    continue
                df = root[key]
                if hasattr(df, "as_polars"):
                    return df.as_polars()
                if isinstance(df, pl.DataFrame):
                    return df
    return None

def _make_bus_index_groups(net: pp.pandapowerNet):
    load_by_bus, sgen_by_bus = {}, {}
    if hasattr(net, "load") and net.load is not None and not net.load.empty:
        for idx, row in net.load.iterrows():
            load_by_bus.setdefault(int(row["bus"]), []).append(int(idx))
    if hasattr(net, "sgen") and net.sgen is not None and not net.sgen.empty:
        for idx, row in net.sgen.iterrows():
            sgen_by_bus.setdefault(int(row["bus"]), []).append(int(idx))
    return load_by_bus, sgen_by_bus

def _assign_vector(df, idxs, col, total_target):
    if not len(idxs):
        return
    baseline = df.loc[idxs, col].astype(float).values
    tot = float(np.sum(baseline))
    frac = (baseline / tot) if tot > 1e-12 else (np.ones(len(idxs)) / len(idxs))
    df.loc[idxs, col] = frac * float(total_target)

def apply_scenario_injections_to_pp_from_nodeedge(
    net_base: pp.pandapowerNet,
    grid_nodeedge,
    scenario: int,
    s_base_w: float = S_BASE_W,
) -> pp.pandapowerNet:
    """Deep-copy net_base and write one NodeEdgeModel scenario into PP net."""
    net_s = _copy.deepcopy(net_base)
    scen_df = _scenario_df_from_nodeedge(grid_nodeedge, scenario)
    if scen_df is None:
        print(f"[WARN] No scenario table on grid_data; scenario {scenario} unchanged.")
        return net_s

    s = scen_df.select([
        c("node_id").cast(pl.Int32),
        pl.coalesce([c("p_cons_pu")]).fill_null(0.0).alias("p_cons_pu"),
        pl.coalesce([c("q_cons_pu")]).fill_null(0.0).alias("q_cons_pu"),
        pl.coalesce([c("p_prod_pu")]).fill_null(0.0).alias("p_prod_pu"),
        pl.coalesce([c("q_prod_pu")]).fill_null(0.0).alias("q_prod_pu"),
    ]).to_pandas()

    load_by_bus, sgen_by_bus = _make_bus_index_groups(net_s)
    pu_to_MW = s_base_w / 1e6
    pu_to_Mvar = s_base_w / 1e6

    for _, row in s.iterrows():
        bus = int(row["node_id"])
        if bus in load_by_bus:
            _assign_vector(net_s.load, load_by_bus[bus], "p_mw",   row["p_cons_pu"] * pu_to_MW)
            _assign_vector(net_s.load, load_by_bus[bus], "q_mvar", row["q_cons_pu"] * pu_to_Mvar)
        if hasattr(net_s, "sgen") and net_s.sgen is not None and not net_s.sgen.empty:
            if bus in sgen_by_bus:
                _assign_vector(net_s.sgen, sgen_by_bus[bus], "p_mw",   row["p_prod_pu"] * pu_to_MW)
                _assign_vector(net_s.sgen, sgen_by_bus[bus], "q_mvar", row["q_prod_pu"] * pu_to_Mvar)
    return net_s

# ----------- LOSSES-ONLY objective (MW) for plotting on y-axis -----------------
def pp_losses_MW(net: pp.pandapowerNet) -> float:
    losses = 0.0
    if hasattr(net, "res_line") and net.res_line is not None and not net.res_line.empty:
        losses += float(net.res_line["pl_mw"].sum())
    if hasattr(net, "res_trafo") and net.res_trafo is not None and not net.res_trafo.empty:
        losses += float(net.res_trafo["pl_mw"].sum())
    return losses

# %% ------------------ Stage A: ADMM on 100 train scenarios -> learn y* --------
net = pp.from_pickle("data/simple_grid.p")

groups = {
    0: [19, 20, 21, 29, 32, 35],
    1: [35, 30, 33, 25, 26, 27],
    2: [27, 32, 22, 23, 34],
    3: [31, 24, 28, 21, 22, 23],
    4: [34, 26, 25, 24, 31],
}

need_train = not (os.path.exists(fp_switch) and os.path.exists(fp_map))

if need_train:
    print("[ADMM][TRAIN] Build train scenarios & solve ADMM to learn y* ...")
    np.random.seed(42)
    grid_train = pandapower_to_dig_a_plan_schema_with_scenarios(
        net, number_of_random_scenarios=NUM_SCEN_TRAIN,
        p_bounds=(-0.6, 1.5), q_bounds=(-0.1, 0.1),
        v_bounds=(-0.1, 0.1), v_min=0.95, v_max=1.05, s_base=S_BASE_W,
        seed=42,
    )
    config_train = ADMMConfig(
        verbose=False, pipeline_type=PipelineType.ADMM, solver_name="gurobi",
        solver_non_convex=2, big_m=1e3, ε=1, ρ=2.0,
        γ_infeasibility=10, γ_admm_penalty=1.0, γ_trafo_loss=1e2,
        groups=groups, max_iters=20, μ=10.0, τ_incr=2.0, τ_decr=2.0,
    )
    dap_train = DigAPlanADMM(config=config_train)
    dap_train.add_grid_data(grid_train)
    dap_train.solve_model()

    zδ_df = dap_train.model_manager.zδ_variable
    zζ_df = dap_train.model_manager.zζ_variable
    z_switch_y = harmonize_by_groups(_switch_df_to_dict(zδ_df), groups, rule="majority")
    tap_choice_y = _taps_df_to_choice(zζ_df)

    edge_to_pp_switch = _build_edge_to_pp_switch_index(dap_train)

    with open(fp_switch, "w") as f: json.dump({int(k): float(v) for k, v in z_switch_y.items()}, f)
    with open(fp_taps,   "w") as f: json.dump({int(k): int(v)   for k, v in tap_choice_y.items()}, f)
    with open(fp_map,    "w") as f: json.dump({int(k): int(v)   for k, v in edge_to_pp_switch.items()}, f)
else:
    print("[CACHE] Loading cached y* and mapping ...")
    with open(fp_switch, "r") as f: z_switch_y = {int(k): float(v) for k, v in json.load(f).items()}
    with open(fp_taps,   "r") as f: tap_choice_y = {int(k): int(v) for k, v in json.load(f).items()}
    with open(fp_map,    "r") as f: edge_to_pp_switch = {int(k): int(v) for k, v in json.load(f).items()}

print("Sample y* (δ):", list(z_switch_y.items())[:5])

# Prepare PP nets (baseline & y*-fixed), annotated with edge_id
annotate_pp_switch_with_edge_ids(net, edge_to_pp_switch)
net_baseline = _copy.deepcopy(net)
net_yfixed   = _copy.deepcopy(net)
annotate_pp_switch_with_edge_ids(net_baseline, edge_to_pp_switch)
annotate_pp_switch_with_edge_ids(net_yfixed,   edge_to_pp_switch)
num_changed = apply_y_to_pandapower_switches(net_yfixed, z_switch_y)
print(f"[DIAG] y* changed {num_changed} PP switches (δ=1->closed: {POLARITY_DELTA_1_IS_CLOSED})")

# Re-create the **same** train scenarios (without re-solving ADMM) for PP evaluation
grid_train_pp = pandapower_to_dig_a_plan_schema_with_scenarios(
    net, number_of_random_scenarios=NUM_SCEN_TRAIN,
    p_bounds=(-0.6, 1.5), q_bounds=(-0.1, 0.1),
    v_bounds=(-0.1, 0.1), v_min=0.95, v_max=1.05, s_base=S_BASE_W,
    seed=42,   
)

# %% --------- PP on TRAIN set: losses for Fixed y* vs Normal-Open --------------
rows_pp_train_fixed, rows_pp_train_no = [], []
for s in range(NUM_SCEN_TRAIN):
    # Fixed y*
    net_s_fixed = apply_scenario_injections_to_pp_from_nodeedge(net_yfixed,   grid_train_pp, scenario=s, s_base_w=S_BASE_W)
    try:
        pp.runpp(net_s_fixed, algorithm="nr")
        loss_fixed = pp_losses_MW(net_s_fixed)
    except Exception:
        loss_fixed = float("inf")
    rows_pp_train_fixed.append({"scenario": s, "loss_MW_PP_FixedY_train": loss_fixed})

    # Normal-Open baseline
    net_s_no = apply_scenario_injections_to_pp_from_nodeedge(net_baseline, grid_train_pp, scenario=s, s_base_w=S_BASE_W)
    try:
        pp.runpp(net_s_no, algorithm="nr")
        loss_no = pp_losses_MW(net_s_no)
    except Exception:
        loss_no = float("inf")
    rows_pp_train_no.append({"scenario": s, "loss_MW_PP_NormalOpen_train": loss_no})

df_PP_train_fixed = pd.DataFrame(rows_pp_train_fixed)
df_PP_train_no    = pd.DataFrame(rows_pp_train_no)
df_PP_train_fixed.to_csv(".cache/figs/pp_losses_train_fixedY.csv", index=False)
df_PP_train_no.to_csv(".cache/figs/pp_losses_train_normalopen.csv", index=False)

# %% ------------------ Stage B: build 100 OOS scenarios ------------------------
print("[OOS] Building OOS scenarios ...")
grid_test = pandapower_to_dig_a_plan_schema_with_scenarios(
    net, number_of_random_scenarios=NUM_SCEN_TEST,
    p_bounds=(-0.6, 1.5), q_bounds=(-0.1, 0.1),
    v_bounds=(-0.1, 0.1), v_min=0.95, v_max=1.05, s_base=S_BASE_W,
    seed=777,
)

# %% --------- PP on OOS set: losses for Fixed y* vs Normal-Open ----------------
rows_pp_oos_fixed, rows_pp_oos_no = [], []
for s in range(NUM_SCEN_TEST):
    # Fixed y*
    net_s_fixed = apply_scenario_injections_to_pp_from_nodeedge(net_yfixed,   grid_test, scenario=s, s_base_w=S_BASE_W)
    try:
        pp.runpp(net_s_fixed, algorithm="nr")
        loss_fixed = pp_losses_MW(net_s_fixed)
    except Exception:
        loss_fixed = float("inf")
    rows_pp_oos_fixed.append({"scenario": s, "loss_MW_PP_FixedY_oos": loss_fixed})

    # Normal-Open
    net_s_no = apply_scenario_injections_to_pp_from_nodeedge(net_baseline, grid_test, scenario=s, s_base_w=S_BASE_W)
    try:
        pp.runpp(net_s_no, algorithm="nr")
        loss_no = pp_losses_MW(net_s_no)
    except Exception:
        loss_no = float("inf")
    rows_pp_oos_no.append({"scenario": s, "loss_MW_PP_NormalOpen_oos": loss_no})

df_PP_oos_fixed = pd.DataFrame(rows_pp_oos_fixed)
df_PP_oos_no    = pd.DataFrame(rows_pp_oos_no)
df_PP_oos_fixed.to_csv(".cache/figs/pp_losses_oos_fixedY.csv", index=False)
df_PP_oos_no.to_csv(".cache/figs/pp_losses_oos_normalopen.csv", index=False)

# %% ------------------ Plots (losses only) -------------------------------------
# Train set comparison
df_train = (df_PP_train_fixed.merge(df_PP_train_no, on="scenario"))
plt.figure(figsize=(9,5))
plt.boxplot(
    [df_train["loss_MW_PP_FixedY_train"].to_numpy(float),
     df_train["loss_MW_PP_NormalOpen_train"].to_numpy(float)],
    labels=["PP: Fixed y* (train)", "PP: Normal-Open (train)"], patch_artist=True, showmeans=True # type: ignore
)
plt.yscale("symlog", linthresh=0.01)
plt.grid(True, axis="y", alpha=0.3)
plt.ylabel("Losses (MW)")
plt.title(f"TRAIN losses across {NUM_SCEN_TRAIN} scenarios")
plt.tight_layout()
plt.savefig(".cache/figs/boxplot_PP_losses_train.svg", bbox_inches="tight")
plt.show()

# OOS comparison
df_oos = (df_PP_oos_fixed.merge(df_PP_oos_no, on="scenario"))
plt.figure(figsize=(9,5))
plt.boxplot(
    [df_oos["loss_MW_PP_FixedY_oos"].to_numpy(float),
     df_oos["loss_MW_PP_NormalOpen_oos"].to_numpy(float)],
    labels=["PP: Fixed y* (OOS)", "PP: Normal-Open (OOS)"], patch_artist=True, showmeans=True #type: ignore
)
plt.yscale("symlog", linthresh=0.01)
plt.grid(True, axis="y", alpha=0.3)
plt.ylabel("Losses (MW)")
plt.title(f"OOS losses across {NUM_SCEN_TEST} scenarios")
plt.tight_layout()
plt.savefig(".cache/figs/boxplot_PP_losses_oos.svg", bbox_inches="tight")
plt.show()

# Optional: save combined tidy table for reporting
df_all = (
    df_train.rename(columns={
        "loss_MW_PP_FixedY_train":"loss_train_fixedY",
        "loss_MW_PP_NormalOpen_train":"loss_train_NO"})
    .merge(df_oos.rename(columns={
        "loss_MW_PP_FixedY_oos":"loss_oos_fixedY",
        "loss_MW_PP_NormalOpen_oos":"loss_oos_NO"}), on="scenario", how="outer")
    .sort_values("scenario").reset_index(drop=True)
)
df_all.to_csv(".cache/figs/pp_losses_train_and_oos.csv", index=False)

# %% ------------------ ONE combined figure (losses, MW) ------------------------
# Prepare the three series
loss_train_fixedY = df_PP_train_fixed["loss_MW_PP_FixedY_train"].to_numpy(float)
loss_oos_fixedY   = df_PP_oos_fixed["loss_MW_PP_FixedY_oos"].to_numpy(float)
loss_train_NO     = df_PP_train_no["loss_MW_PP_NormalOpen_train"].to_numpy(float)

plt.figure(figsize=(10,5))
plt.boxplot(
    [loss_train_fixedY, loss_oos_fixedY, loss_train_NO],
    labels=["PP: Fixed y* (TRAIN)", "PP: Fixed y* (OOS)", "PP: Normal-Open (TRAIN)"], #type: ignore
    patch_artist=True,
    showmeans=True
)
plt.yscale("symlog", linthresh=0.01)   # keeps small values visible, compresses large ones
plt.grid(True, axis="y", alpha=0.3)
plt.ylabel("Losses (MW)")
plt.title(
    f"Pandapower losses: TRAIN Fixed y* vs OOS Fixed y* vs TRAIN Normal-Open\n"
    f"(train N={len(loss_train_fixedY)}, oos N={len(loss_oos_fixedY)})"
)
plt.tight_layout()
os.makedirs(".cache/figs", exist_ok=True)
plt.savefig(".cache/figs/boxplot_PP_losses_train_fixedY__oos_fixedY__train_NO.svg",
            bbox_inches="tight")
plt.show()


# %%

# %%
