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
# ===============================================================
LIMIT = 100.0
STEP = 20.0
MAX_ROUNDS = 30  # safety stop

results = []
logs_by_scenario = {}

for scen_key, scen_df in rand_scenarios.items():
    net_case = apply_profile_scenario_to_pandapower(net0, scen_df, grid.s_base)

    reinforced_lines_all = set()
    reinforced_trafos_all = set()

    round_log = []

    converged = False
    for r in range(MAX_ROUNDS):
        # 1) PF
        try:
            pp.runpp(net_case, check_connectivity=True, init="auto")
        except Exception as e:
            round_log.append({
                "round": r,
                "status": f"PF failed: {e}",
                "n_congested_lines": None,
                "n_congested_trafos": None,
                "max_line_loading": None,
                "max_trafo_loading": None,
            })
            break

        # 2) detect congestion
        cong_lines = check_line_loading(net_case, limit_percent=LIMIT)
        cong_trafos = check_trafo_loading(net_case, limit_percent=LIMIT)

        nL = len(cong_lines)
        nT = len(cong_trafos)

        max_line = net_case.res_line["loading_percent"].max() if len(net_case.res_line) else np.nan
        max_trafo = net_case.res_trafo["loading_percent"].max() if len(net_case.res_trafo) else np.nan

        round_log.append({
            "round": r,
            "status": "OK",
            "n_congested_lines": nL,
            "n_congested_trafos": nT,
            "max_line_loading": float(max_line) if max_line == max_line else np.nan,
            "max_trafo_loading": float(max_trafo) if max_trafo == max_trafo else np.nan,
        })

        # 3) stop condition
        if nL == 0 and nT == 0:
            converged = True
            break

        # 4) reinforce congested assets by +20%
        if nL > 0:
            for lid in cong_lines["line_idx"].tolist():
                lid = int(lid)
                reinforced_lines_all.add(lid)
                reinforce_line_one_step(net_case, lid, step_percent=STEP)

        if nT > 0:
            for tid in cong_trafos["trafo_idx"].tolist():
                tid = int(tid)
                reinforced_trafos_all.add(tid)
                reinforce_trafo_one_step(net_case, tid, step_percent=STEP)

    # final PF (optional, ensures results are fresh if converged)
    if converged:
        pp.runpp(net_case, check_connectivity=True, init="auto")

    final_cong_lines = check_line_loading(net_case, limit_percent=LIMIT)
    final_cong_trafos = check_trafo_loading(net_case, limit_percent=LIMIT)

    results.append({
        "scenario": scen_key,
        "converged_to_zero": converged,
        "rounds_used": len(round_log),
        "final_n_congested_lines": len(final_cong_lines),
        "final_n_congested_trafos": len(final_cong_trafos),
        "final_max_line_loading": net_case.res_line["loading_percent"].max() if len(net_case.res_line) else np.nan,
        "final_max_trafo_loading": net_case.res_trafo["loading_percent"].max() if len(net_case.res_trafo) else np.nan,
        "reinforced_lines": ",".join(map(str, sorted(reinforced_lines_all))),
        "reinforced_trafos": ",".join(map(str, sorted(reinforced_trafos_all))),
    })

    logs_by_scenario[scen_key] = pd.DataFrame(round_log)

    print(
        f"Scenario {scen_key}: converged={converged}, "
        f"rounds={len(round_log)}, "
        f"final cong lines={len(final_cong_lines)}, "
        f"final cong trafos={len(final_cong_trafos)}"
    )

summary_df = pd.DataFrame(results)
display(summary_df)  # type: ignore