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
SCEN_COUNTS = [50, 100, 150]   # training sizes
S_BASE_W    = 1e6
TRAIN_SEED  = 42
OOS_SEED    = 777
OOS_N       = 500              # out-of-sample scenarios
USE_CACHE   = False

# δ polarity:
# True  => δ=1 means CLOSED; δ=0 means OPEN
# False => δ=1 means OPEN;   δ=0 means CLOSED
POLARITY_DELTA_1_IS_CLOSED = True

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
            best[int(eid)] = tap  # type: ignore
        return best
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

# =============== robust mapping using eq_fk ('switch i') ==================
def _pp_switch_index_by_name(net: pp.pandapowerNet) -> dict[str, int]:
    """Build {'switch 0': row_index, ...} from net.switch['name'] (case-insensitive)."""
    if not hasattr(net, "switch") or net.switch is None or net.switch.empty:
        return {}
    if "name" not in net.switch.columns:
        raise RuntimeError("net.switch has no 'name' column to match eq_fk.")
    name_to_idx = {}
    for idx, row in net.switch.iterrows():
        nm = str(row["name"]).strip().lower()
        name_to_idx[nm] = int(idx)
    return name_to_idx

def build_edge_to_pp_switch_index_from_dap_and_net(dap, net: pp.pandapowerNet) -> dict[int, int]:
    """Use ADMM zδ_variable (eq_fk, edge_id) + PP net.switch['name'] to build: edge_id -> net.switch index."""
    zdelta: pl.DataFrame = dap.model_manager.zδ_variable
    if "eq_fk" not in zdelta.columns or "edge_id" not in zdelta.columns:
        raise RuntimeError("zδ_variable must have 'eq_fk' and 'edge_id' columns.")
    name_to_idx = _pp_switch_index_by_name(net)
    mapping: dict[int, int] = {}
    for r in zdelta.select(["eq_fk", "edge_id"]).to_dicts():
        eq_fk = str(r["eq_fk"]).strip().lower()  
        eid   = int(r["edge_id"])                
        if eq_fk in name_to_idx:
            mapping[eid] = name_to_idx[eq_fk]
    if not mapping:
        raise RuntimeError("Could not build any edge_id -> net.switch index mapping (name mismatch?).")
    return mapping

def annotate_pp_switch_with_edge_ids_from_mapping(net: pp.pandapowerNet, edge_to_pp_switch: dict[int, int]) -> None:
    """Create/overwrite net.switch['edge_id'] using a provided {edge_id -> pp_index} mapping."""
    if not hasattr(net, "switch") or net.switch is None or net.switch.empty:
        return
    if "edge_id" not in net.switch.columns:
        net.switch["edge_id"] = -1
    net.switch.loc[:, "edge_id"] = -1
    for eid, pp_idx in edge_to_pp_switch.items():
        if pp_idx in net.switch.index:
            net.switch.at[pp_idx, "edge_id"] = int(eid)
    net.switch["edge_id"] = net.switch["edge_id"].astype(int)

def apply_y_to_pandapower_switches(net: pp.pandapowerNet, z_switch_map: dict[int, float]) -> int:
    """Write net.switch.closed by matching on net.switch['edge_id']. Returns #changed."""
    if not hasattr(net, "switch") or net.switch is None or net.switch.empty:
        return 0
    if "edge_id" not in net.switch.columns:
        raise RuntimeError("net.switch has no 'edge_id' column; annotate it first.")
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

def apply_normal_open_to_pp_switches(net: pp.pandapowerNet,
                                     normal_open_map: dict[int, bool]) -> int:
    """Force PP net.switch.closed to the ADMM 'normal_open' state: closed = not normal_open."""
    if not hasattr(net, "switch") or net.switch is None or net.switch.empty:
        return 0
    if "edge_id" not in net.switch.columns:
        raise RuntimeError("net.switch has no 'edge_id'. Annotate it first.")
    changed = 0
    for eid, no in normal_open_map.items():
        mask = (net.switch["edge_id"] == int(eid))
        if not mask.any():
            continue
        desired_closed = not bool(no)
        before = net.switch.loc[mask, "closed"].astype(bool)
        net.switch.loc[mask, "closed"] = desired_closed
        changed += int((before.values != desired_closed).sum())
    return changed

# =============== PANDAPOWER injections & losses ====================
def _scenario_df_from_nodeedge(grid_nodeedge, scenario: int) -> pl.DataFrame | None:
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
    pu_to_MW   = s_base_w / 1e6
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

def pp_losses_MW(net: pp.pandapowerNet) -> float:
    losses = 0.0
    if hasattr(net, "res_line") and net.res_line is not None and not net.res_line.empty:
        losses += float(net.res_line["pl_mw"].sum())
    if hasattr(net, "res_trafo") and net.res_trafo is not None and not net.res_trafo.empty:
        losses += float(net.res_trafo["pl_mw"].sum())
    return losses

# %%# -------------------- run for N in {50,100,150} --------------------
net_base = pp.from_pickle("data/simple_grid.p")

groups = {
    0: [19, 20, 21, 29, 32, 35],
    1: [35, 30, 33, 25, 26, 27],
    2: [27, 32, 22, 23, 34],
    3: [31, 24, 28, 21, 22, 23],
    4: [34, 26, 25, 24, 31],
}

# containers for plotting
dist_train_fixedY = {}   # N -> np.array (size N)
dist_oos_fixedY   = {}   # N -> np.array (size OOS_N)
dist_train_NO     = {}   # N -> np.array (size N)
dist_oos_NO       = {}   # N -> np.array (size OOS_N)

for N in SCEN_COUNTS:
    SAVE_DIR = f".cache/admm_insample_{N}"
    os.makedirs(SAVE_DIR, exist_ok=True)
    fp_switch = os.path.join(SAVE_DIR, "consensus_switch.json")
    fp_taps   = os.path.join(SAVE_DIR, "consensus_taps.json")
    fp_map    = os.path.join(SAVE_DIR, "edge_to_pp_switch.json")

    net = _copy.deepcopy(net_base)

    need_train = True
    if USE_CACHE and all(os.path.exists(p) for p in [fp_switch, fp_taps, fp_map]):
        need_train = False

    if need_train:
        print(f"\n[ADMM][TRAIN][N={N}] Building {N} train scenarios & solving...")
        np.random.seed(TRAIN_SEED)
        grid_train = pandapower_to_dig_a_plan_schema_with_scenarios(
            net, number_of_random_scenarios=N,
            p_bounds=(-0.6, 1.5), q_bounds=(-0.1, 0.1),
            v_bounds=(-0.1, 0.1), v_min=0.95, v_max=1.05, s_base=S_BASE_W,
            seed=TRAIN_SEED,
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

        edge_to_pp_switch = build_edge_to_pp_switch_index_from_dap_and_net(dap_train, net)
        with open(fp_map, "w") as f: json.dump({int(k): int(v) for k, v in edge_to_pp_switch.items()}, f)
        with open(fp_switch, "w") as f: json.dump({int(k): float(v) for k, v in z_switch_y.items()}, f)
        with open(fp_taps,   "w") as f: json.dump({int(k): int(v)   for k, v in tap_choice_y.items()}, f)
    else:
        print(f"\n[CACHE][N={N}] Loading cached y* and mapping ...")
        with open(fp_switch, "r") as f: z_switch_y = {int(k): float(v) for k, v in json.load(f).items()}
        with open(fp_taps,   "r") as f: tap_choice_y = {int(k): int(v) for k, v in json.load(f).items()}
        with open(fp_map,    "r") as f: edge_to_pp_switch = {int(k): int(v) for k, v in json.load(f).items()}

    # Build Normal-Open by re-solving with fixed switches (uses same TRAIN scenarios)
    if not (USE_CACHE and os.path.exists(os.path.join(SAVE_DIR, "normal_open_map.json"))):
        # need dap_train + grid_train; rebuild minimal grid_train for consistency
        grid_train_for_NO = pandapower_to_dig_a_plan_schema_with_scenarios(
            net, number_of_random_scenarios=N,
            p_bounds=(-0.6, 1.5), q_bounds=(-0.1, 0.1),
            v_bounds=(-0.1, 0.1), v_min=0.95, v_max=1.05, s_base=S_BASE_W,
            seed=TRAIN_SEED,
        )
        config_train_NO = ADMMConfig(
            verbose=False, pipeline_type=PipelineType.ADMM, solver_name="gurobi",
            solver_non_convex=2, big_m=1e3, ε=1, ρ=2.0,
            γ_infeasibility=10, γ_admm_penalty=1.0, γ_trafo_loss=1e2,
            groups=groups, max_iters=20, μ=10.0, τ_incr=2.0, τ_decr=2.0,
        )
        dap_train_for_NO = DigAPlanADMM(config=config_train_NO)
        dap_train_for_NO.add_grid_data(grid_train_for_NO)
        dap_train_for_NO.solve_model(fixed_switches=True)
        zδ_df_fixed = dap_train_for_NO.model_manager.zδ_variable
        use_col = "normal_open" if ("normal_open" in zδ_df_fixed.columns and
                                    bool(zδ_df_fixed.select(pl.col("normal_open").cast(pl.Int8).sum()).item())) else "open"
        normal_open_map = {int(r["edge_id"]): bool(r[use_col]) for r in zδ_df_fixed.select(["edge_id", use_col]).to_dicts()}
        with open(os.path.join(SAVE_DIR, "normal_open_map.json"), "w") as f:
            json.dump({int(k): bool(v) for k, v in normal_open_map.items()}, f)
    else:
        with open(os.path.join(SAVE_DIR, "normal_open_map.json"), "r") as f:
            normal_open_map = {int(k): bool(v) for k, v in json.load(f).items()}

    # Prepare PP nets and annotate edge_id
    net_yfixed = _copy.deepcopy(net)
    net_NO     = _copy.deepcopy(net)
    annotate_pp_switch_with_edge_ids_from_mapping(net_yfixed, edge_to_pp_switch)
    annotate_pp_switch_with_edge_ids_from_mapping(net_NO,     edge_to_pp_switch)

    _ = apply_y_to_pandapower_switches(net_yfixed, z_switch_y)
    _ = apply_normal_open_to_pp_switches(net_NO, normal_open_map)

    # TRAIN scenarios (N of them) -> evaluate PP losses
    grid_train_pp = pandapower_to_dig_a_plan_schema_with_scenarios(
        net, number_of_random_scenarios=N,
        p_bounds=(-0.6, 1.5), q_bounds=(-0.1, 0.1),
        v_bounds=(-0.1, 0.1), v_min=0.95, v_max=1.05, s_base=S_BASE_W,
        seed=TRAIN_SEED,
    )

    tr_fixed_losses, tr_no_losses = [], []
    for s in range(N):
        # Fixed y*
        net_s_fixed = apply_scenario_injections_to_pp_from_nodeedge(net_yfixed, grid_train_pp, scenario=s, s_base_w=S_BASE_W)
        try:
            pp.runpp(net_s_fixed, algorithm="nr")
            tr_fixed_losses.append(pp_losses_MW(net_s_fixed))
        except Exception:
            tr_fixed_losses.append(float("inf"))

        # Normal-Open
        net_s_no = apply_scenario_injections_to_pp_from_nodeedge(net_NO, grid_train_pp, scenario=s, s_base_w=S_BASE_W)
        try:
            pp.runpp(net_s_no, algorithm="nr")
            tr_no_losses.append(pp_losses_MW(net_s_no))
        except Exception:
            tr_no_losses.append(float("inf"))

    dist_train_fixedY[N] = np.asarray(tr_fixed_losses, dtype=float)
    dist_train_NO[N]     = np.asarray(tr_no_losses,    dtype=float)

    # OOS scenarios (N of them) -> evaluate PP losses for Fixed y*
    grid_test = pandapower_to_dig_a_plan_schema_with_scenarios(
        net, number_of_random_scenarios=OOS_N,
        p_bounds=(-0.6, 1.5), q_bounds=(-0.1, 0.1),
        v_bounds=(-0.1, 0.1), v_min=0.95, v_max=1.05, s_base=S_BASE_W,
        seed=OOS_SEED,
    )

    oos_fixed_losses = []
    oos_no_losses    = []   
    for s in range(N):
        net_s_fixed = apply_scenario_injections_to_pp_from_nodeedge(net_yfixed, grid_test, scenario=s, s_base_w=S_BASE_W)
        try:
            pp.runpp(net_s_fixed, algorithm="nr")
            oos_fixed_losses.append(pp_losses_MW(net_s_fixed))
        except Exception:
            oos_fixed_losses.append(float("inf"))
            
        net_s_no = apply_scenario_injections_to_pp_from_nodeedge(net_NO, grid_test, scenario=s, s_base_w=S_BASE_W)
        try:
            pp.runpp(net_s_no, algorithm="nr")
            oos_no_losses.append(pp_losses_MW(net_s_no))
        except Exception:
            oos_no_losses.append(float("inf"))

    dist_oos_fixedY[N] = np.asarray(oos_fixed_losses, dtype=float)
    dist_oos_NO[N]     = np.asarray(oos_no_losses,    dtype=float)

# %% # -------------------- Plotting --------------------

# -------------------- Single figure: TRAIN vs OOS=500 + one Normal-Open=500 ---
os.makedirs(".cache/figs", exist_ok=True)

# pick ONE Normal-Open@500 series (here: from the last training N)
no_key = SCEN_COUNTS[-1]
no_500 = dist_oos_NO[no_key]  # uses OOS_N=500, seed 777

plt.figure(figsize=(12,6))
positions, data, labels = [], [], []

# layout params
group_gap = 0.9     # gap between N groups
box_w     = 0.28    # width of each box
base      = 1.0

# add 2 boxes (TRAIN N, OOS 500) for each N
for i, N in enumerate(SCEN_COUNTS):
    pos_train = base + i*(2*box_w + group_gap)
    pos_oos   = pos_train + box_w

    positions.extend([pos_train, pos_oos])
    data.extend([dist_train_fixedY[N], dist_oos_fixedY[N]])
    labels.extend([f"In-Sample\nN={N}", f"OOS\nN=500"])

# add ONE Normal-Open (OOS=500) box at the end
pos_no = (base + len(SCEN_COUNTS)*(2*box_w + group_gap))
positions.append(pos_no)
data.append(no_500)
labels.append("Normal-Open\nN=500")

plt.boxplot(data, positions=positions, widths=box_w, patch_artist=True, showmeans=True)
plt.yscale("symlog", linthresh=0.01)
plt.grid(True, axis="y", alpha=0.3)
plt.xticks(positions, labels)
plt.ylabel("Losses (p.u.)")
#plt.title("PP losses — Fixed y* (TRAIN vs OOS=500) and one Normal-Open (OOS=500, seed=777)")
plt.tight_layout()
plt.savefig(".cache/figs/fig_single_box_train_vs_oos500_plus_normalopen.svg", bbox_inches="tight")
plt.show()

# %%
