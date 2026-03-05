# %% ============================================================
# 1) Imports
# ===============================================================
from pathlib import Path
import json
import copy

import numpy as np
import pandas as pd
import pandapower as pp
import polars as pl

from api.grid_cases import generate_profile_based_load_scenarios
from data_model import GridCaseModel, ShortTermUncertaintyProfile
from data_model.kace import (
    DiscreteScenario,
)
from data_exporter.pp_to_dap import pp_to_dap
from experiments.notebook.Scenario_PP import apply_profile_scenario_to_pandapower
from experiments.notebook.congestion_helpers import (
    check_line_loading, check_trafo_loading,
    check_voltage_violations,
    reinforce_line_one_step,
    reinforce_trafo_one_step,
)

# %% ============================================================
# 2) Load JSON config file
# ===============================================================
project_root = Path.cwd().parent
json_file = project_root / "experiments" / "notebook" / "00-power-flow.json"

print("Resolved path:", json_file.resolve())
print("Exists:", json_file.exists())

with open(json_file, "r", encoding="utf-8") as f:
    config = json.load(f)

print("Loaded JSON content:")
print(json.dumps(config, indent=2, ensure_ascii=False))

# %% ============================================================
# 4) Extract and normalize config parameters
# ===============================================================
grid_cfg = config["grid"].copy()
profiles_cfg = config["profiles"].copy()
seed = int(config["seed"])

# ---- Make paths absolute (relative to project root) ----
grid_cfg["pp_file"] = str(project_root / grid_cfg["pp_file"])
grid_cfg["egid_id_mapping_file"] = str(project_root / grid_cfg["egid_id_mapping_file"])

profiles_cfg["load_profiles"] = [str(project_root / p) for p in profiles_cfg["load_profiles"]]
profiles_cfg["pv_profile"] = str(project_root / profiles_cfg["pv_profile"])

# ---- Convert scenario_name string -> DiscreteScenario Enum ----
profiles_cfg["scenario_name"] = DiscreteScenario(profiles_cfg["scenario_name"])

# ---- Create Pydantic objects ----
grid = GridCaseModel(**grid_cfg)
profiles = ShortTermUncertaintyProfile(**profiles_cfg)

# %% ============================================================
# 6) Load the pandapower network
# ===============================================================
net0 = pp.from_pickle(grid.pp_file)

# %% ============================================================
# 7) Generate profile-based scenarios
# ===============================================================
rand_scenarios = generate_profile_based_load_scenarios(
    grid=grid,
    profiles=profiles,
    net=net0,
    seed=seed,
)

# %% ============================================================
# Reinforce iteratively (20% step) until congestion is zero
# + compute congestion rate BEFORE reinforcement
# + compute reinforcement cost when congestion becomes zero
# ===============================================================
LIMIT = 100.0
STEP = 20.0
MAX_ROUNDS = 30  # safety stop

# ---- cost parameters (edit to your values/units) ----
COST_PER_KA_KM = 50_000.0   # CHF per (kA * km) added ampacity
COST_PER_MVA   = 20_000.0   # CHF per MVA added transformer capacity

results = []
logs_by_scenario = {}

for scen_key, scen_df in rand_scenarios.items():
    net_case = apply_profile_scenario_to_pandapower(net0, scen_df, grid.s_base)

    # Store initial capacities to compute deltas/cost at the end
    line_max_i_init = net_case.line["max_i_ka"].copy() if len(net_case.line) else pd.Series(dtype=float)
    trafo_sn_init   = net_case.trafo["sn_mva"].copy() if len(net_case.trafo) else pd.Series(dtype=float)

    # ---------- PF BEFORE any reinforcement ----------
    pp.runpp(net_case, check_connectivity=True, init="auto")

    # Congestion BEFORE reinforcement (rate + max loading)
    cong_lines_before = check_line_loading(net_case, limit_percent=LIMIT)
    cong_trafos_before = check_trafo_loading(net_case, limit_percent=LIMIT)

    n_lines_total = len(net_case.line)
    n_trafos_total = len(net_case.trafo)

    cong_rate_lines_before = (len(cong_lines_before) / n_lines_total) if n_lines_total > 0 else 0.0
    cong_rate_trafos_before = (len(cong_trafos_before) / n_trafos_total) if n_trafos_total > 0 else 0.0

    max_line_before = net_case.res_line["loading_percent"].max() if len(net_case.res_line) else np.nan
    max_trafo_before = net_case.res_trafo["loading_percent"].max() if len(net_case.res_trafo) else np.nan

    # Track which assets were ever reinforced
    reinforced_lines_all = set()
    reinforced_trafos_all = set()

    round_log = []
    converged = False

    for r in range(MAX_ROUNDS):
        # 1) PF
        pp.runpp(net_case, check_connectivity=True, init="auto")

        # 2) detect congestion
        cong_lines = check_line_loading(net_case, limit_percent=LIMIT)
        cong_trafos = check_trafo_loading(net_case, limit_percent=LIMIT)

        nL = len(cong_lines)
        nT = len(cong_trafos)

        max_line = net_case.res_line["loading_percent"].max() if len(net_case.res_line) else np.nan
        max_trafo = net_case.res_trafo["loading_percent"].max() if len(net_case.res_trafo) else np.nan

        round_log.append({
            "round": r,
            "n_congested_lines": nL,
            "n_congested_trafos": nT,
            "max_line_loading": float(max_line) if max_line == max_line else np.nan,
            "max_trafo_loading": float(max_trafo) if max_trafo == max_trafo else np.nan,
        })

        # 3) stop if zero congestion
        if nL == 0 and nT == 0:
            converged = True
            break

        # 4) reinforce congested assets by +20%
        for lid in cong_lines["line_idx"].tolist():
            lid = int(lid)
            reinforced_lines_all.add(lid)
            reinforce_line_one_step(net_case, lid, step_percent=STEP)

        for tid in cong_trafos["trafo_idx"].tolist():
            tid = int(tid)
            reinforced_trafos_all.add(tid)
            reinforce_trafo_one_step(net_case, tid, step_percent=STEP)

    # Final PF to refresh results after last reinforcement
    pp.runpp(net_case, check_connectivity=True, init="auto")

    final_cong_lines = check_line_loading(net_case, limit_percent=LIMIT)
    final_cong_trafos = check_trafo_loading(net_case, limit_percent=LIMIT)

    final_max_line = net_case.res_line["loading_percent"].max() if len(net_case.res_line) else np.nan
    final_max_trafo = net_case.res_trafo["loading_percent"].max() if len(net_case.res_trafo) else np.nan

    # ---------- COST (based on total capacity increases) ----------
    # Lines: cost ∝ Δmax_i_ka * length_km
    if len(net_case.line) > 0:
        delta_i = (net_case.line["max_i_ka"] - line_max_i_init).fillna(0.0)
        lengths = (
            net_case.line["length_km"].fillna(1.0)
            if "length_km" in net_case.line.columns
            else pd.Series(1.0, index=net_case.line.index)
        )
        cost_lines = float((delta_i * lengths * COST_PER_KA_KM).sum())
    else:
        cost_lines = 0.0

    # Trafos: cost ∝ Δsn_mva
    if len(net_case.trafo) > 0:
        delta_sn = (net_case.trafo["sn_mva"] - trafo_sn_init).fillna(0.0)
        cost_trafos = float((delta_sn * COST_PER_MVA).sum())
    else:
        cost_trafos = 0.0

    cost_total = cost_lines + cost_trafos

    # Store results
    results.append({
        "scenario": scen_key,
        "converged_to_zero": bool(converged and len(final_cong_lines) == 0 and len(final_cong_trafos) == 0),
        "rounds_used": len(round_log),

        # congestion BEFORE reinforcement (rate + max)
        "n_congested_lines_before": len(cong_lines_before),
        "n_congested_trafos_before": len(cong_trafos_before),
        "cong_rate_lines_before": cong_rate_lines_before,
        "cong_rate_trafos_before": cong_rate_trafos_before,
        "max_line_loading_before": max_line_before,
        "max_trafo_loading_before": max_trafo_before,

        # final congestion
        "final_n_congested_lines": len(final_cong_lines),
        "final_n_congested_trafos": len(final_cong_trafos),
        "final_max_line_loading": final_max_line,
        "final_max_trafo_loading": final_max_trafo,

        # reinforcement summary
        "reinforced_lines": ",".join(map(str, sorted(reinforced_lines_all))),
        "reinforced_trafos": ",".join(map(str, sorted(reinforced_trafos_all))),

        # costs
        "cost_lines": cost_lines,
        "cost_trafos": cost_trafos,
        "cost_total": cost_total,
    })

    logs_by_scenario[scen_key] = pd.DataFrame(round_log)

    print(
        f"Scenario {scen_key}: "
        f"rate_before(lines)={cong_rate_lines_before:.3f}, rate_before(trafos)={cong_rate_trafos_before:.3f}, "
        f"converged={converged}, rounds={len(round_log)}, "
        f"final_cong_lines={len(final_cong_lines)}, final_cong_trafos={len(final_cong_trafos)}, "
        f"cost_total={cost_total:,.2f}"
    )

summary_df = pd.DataFrame(results)
display(summary_df)  # type: ignore