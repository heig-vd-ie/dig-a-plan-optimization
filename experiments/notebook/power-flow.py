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
# 8) Test helpers on all scenarios + one-step reinforcement
# ===============================================================
LIMIT = 100.0
VMIN = 0.95
VMAX = 1.05
STEP = 20.0

results = []
details = {}  # store detailed outputs per scenario (optional)

for scen_key, scen_df in rand_scenarios.items():
    # 1) Apply scenario -> pandapower net
    net_case = apply_profile_scenario_to_pandapower(net0, scen_df, grid.s_base)

    # 2) Run PF (BEFORE)
    try:
        pp.runpp(net_case, check_connectivity=True, init="auto")
        pf_before_ok = True
    except Exception as e:
        pf_before_ok = False
        results.append({
            "scenario": scen_key,
            "status": f"PF failed before: {e}",
            "max_line_loading_before": np.nan,
            "max_trafo_loading_before": np.nan,
            "min_vm_pu_before": np.nan,
            "max_vm_pu_before": np.nan,
            "n_congested_lines_before": np.nan,
            "n_congested_trafos_before": np.nan,
            "n_undervoltage_before": np.nan,
            "n_overvoltage_before": np.nan,
            "reinforced_lines": "",
            "reinforced_trafos": "",
            "max_line_loading_after": np.nan,
            "max_trafo_loading_after": np.nan,
            "min_vm_pu_after": np.nan,
            "max_vm_pu_after": np.nan,
            "n_congested_lines_after": np.nan,
            "n_congested_trafos_after": np.nan,
            "n_undervoltage_after": np.nan,
            "n_overvoltage_after": np.nan,
        })
        continue

    # 3) Use helper functions (BEFORE)
    ranked_lines_before = check_line_loading(net_case, limit_percent=None)
    ranked_trafos_before = check_trafo_loading(net_case, limit_percent=None)

    congested_lines_before = check_line_loading(net_case, limit_percent=LIMIT)
    congested_trafos_before = check_trafo_loading(net_case, limit_percent=LIMIT)

    uv_before, ov_before = check_voltage_violations(net_case, vmin_pu=VMIN, vmax_pu=VMAX)

    max_line_before = net_case.res_line["loading_percent"].max() if len(net_case.res_line) else np.nan
    max_trafo_before = net_case.res_trafo["loading_percent"].max() if len(net_case.res_trafo) else np.nan
    min_vm_before = net_case.res_bus["vm_pu"].min() if len(net_case.res_bus) else np.nan
    max_vm_before = net_case.res_bus["vm_pu"].max() if len(net_case.res_bus) else np.nan

    # 4) Reinforce congested assets ONCE
    reinforced_line_ids = congested_lines_before["line_idx"].tolist() if "line_idx" in congested_lines_before.columns else []
    reinforced_trafo_ids = congested_trafos_before["trafo_idx"].tolist() if "trafo_idx" in congested_trafos_before.columns else []

    for lid in reinforced_line_ids:
        reinforce_line_one_step(net_case, int(lid), step_percent=STEP)

    for tid in reinforced_trafo_ids:
        reinforce_trafo_one_step(net_case, int(tid), step_percent=STEP)

    # 5) Run PF (AFTER)
    try:
        pp.runpp(net_case, check_connectivity=True, init="auto")
        pf_after_ok = True
    except Exception as e:
        pf_after_ok = False
        results.append({
            "scenario": scen_key,
            "status": f"PF failed after: {e}",
            "max_line_loading_before": max_line_before,
            "max_trafo_loading_before": max_trafo_before,
            "min_vm_pu_before": min_vm_before,
            "max_vm_pu_before": max_vm_before,
            "n_congested_lines_before": len(congested_lines_before),
            "n_congested_trafos_before": len(congested_trafos_before),
            "n_undervoltage_before": len(uv_before),
            "n_overvoltage_before": len(ov_before),
            "reinforced_lines": ",".join(map(str, reinforced_line_ids)),
            "reinforced_trafos": ",".join(map(str, reinforced_trafo_ids)),
            "max_line_loading_after": np.nan,
            "max_trafo_loading_after": np.nan,
            "min_vm_pu_after": np.nan,
            "max_vm_pu_after": np.nan,
            "n_congested_lines_after": np.nan,
            "n_congested_trafos_after": np.nan,
            "n_undervoltage_after": np.nan,
            "n_overvoltage_after": np.nan,
        })
        continue

    # 6) Use helper functions (AFTER)
    ranked_lines_after = check_line_loading(net_case, limit_percent=None)
    ranked_trafos_after = check_trafo_loading(net_case, limit_percent=None)

    congested_lines_after = check_line_loading(net_case, limit_percent=LIMIT)
    congested_trafos_after = check_trafo_loading(net_case, limit_percent=LIMIT)

    uv_after, ov_after = check_voltage_violations(net_case, vmin_pu=VMIN, vmax_pu=VMAX)

    max_line_after = net_case.res_line["loading_percent"].max() if len(net_case.res_line) else np.nan
    max_trafo_after = net_case.res_trafo["loading_percent"].max() if len(net_case.res_trafo) else np.nan
    min_vm_after = net_case.res_bus["vm_pu"].min() if len(net_case.res_bus) else np.nan
    max_vm_after = net_case.res_bus["vm_pu"].max() if len(net_case.res_bus) else np.nan

    # 7) Save summary row
    results.append({
        "scenario": scen_key,
        "status": "OK",
        "max_line_loading_before": max_line_before,
        "max_trafo_loading_before": max_trafo_before,
        "min_vm_pu_before": min_vm_before,
        "max_vm_pu_before": max_vm_before,
        "n_congested_lines_before": len(congested_lines_before),
        "n_congested_trafos_before": len(congested_trafos_before),
        "n_undervoltage_before": len(uv_before),
        "n_overvoltage_before": len(ov_before),
        "reinforced_lines": ",".join(map(str, reinforced_line_ids)),
        "reinforced_trafos": ",".join(map(str, reinforced_trafo_ids)),
        "max_line_loading_after": max_line_after,
        "max_trafo_loading_after": max_trafo_after,
        "min_vm_pu_after": min_vm_after,
        "max_vm_pu_after": max_vm_after,
        "n_congested_lines_after": len(congested_lines_after),
        "n_congested_trafos_after": len(congested_trafos_after),
        "n_undervoltage_after": len(uv_after),
        "n_overvoltage_after": len(ov_after),
    })

    # 8) Store detailed tables (optional, helpful for debugging/reporting)
    details[scen_key] = {
        "ranked_lines_before": ranked_lines_before,
        "ranked_trafos_before": ranked_trafos_before,
        "congested_lines_before": congested_lines_before,
        "congested_trafos_before": congested_trafos_before,
        "uv_before": uv_before,
        "ov_before": ov_before,
        "ranked_lines_after": ranked_lines_after,
        "ranked_trafos_after": ranked_trafos_after,
        "congested_lines_after": congested_lines_after,
        "congested_trafos_after": congested_trafos_after,
        "uv_after": uv_after,
        "ov_after": ov_after,
    }

    print(
        f"Scenario {scen_key}: "
        f"maxLineBefore={max_line_before:.2f}%, "
        f"minVBefore={min_vm_before:.4f}, "
        f"congLinesBefore={len(congested_lines_before)}, "
        f"congTrafosBefore={len(congested_trafos_before)} -> "
        f"maxLineAfter={max_line_after:.2f}%, "
        f"minVAfter={min_vm_after:.4f}, "
        f"congLinesAfter={len(congested_lines_after)}, "
        f"congTrafosAfter={len(congested_trafos_after)}"
    )

summary_df = pd.DataFrame(results)
print("Final summary:")
display(summary_df)  # type: ignore