# %% Import libraries
from pathlib import Path
import json
import copy

import numpy as np
import pandas as pd
import pandapower as pp
import polars as pl

from data_model import GridCaseModel, ShortTermUncertaintyProfile
from data_model.kace import DiscreteScenario
from experiments.reinforcement_power_flow.scenario_pp import (apply_profile_scenario_to_pandapower
, build_snapshot_from_wide_profile)
from experiments.reinforcement_power_flow.congestion_helpers import (
    check_line_loading, check_trafo_loading,
    reinforce_line_one_step,
    reinforce_trafo_one_step,
)

# %% input parameters for reinforcement and congestion settings
LIMIT = 60.0
STEP = 20.0
MAX_ROUNDS = 50  
LINE_COST_PER_KM_KW = 0.2
TRAFO_COST_PER_KW = 0.15

# %% Load JSON config 
project_root = Path.cwd().parent
json_file = project_root / "experiments" / "reinforcement_power_flow" / "00-power-flow.json"

with open(json_file, "r", encoding="utf-8") as f:
    config = json.load(f)

# %% Extract and normalize config parameters
grid_cfg = config["grid"].copy()
profiles_cfg = config["profiles"].copy()

# Make all paths from project root 
grid_cfg["pp_file"] = str(project_root / grid_cfg["pp_file"])
grid_cfg["egid_id_mapping_file"] = str(project_root / grid_cfg["egid_id_mapping_file"])

profiles_cfg["load_profiles"] = [str(project_root / p) for p in profiles_cfg["load_profiles"]]
profiles_cfg["pv_profile"] = str(project_root / profiles_cfg["pv_profile"])
profiles_cfg["scenario_name"] = DiscreteScenario(profiles_cfg["scenario_name"])
stage_years = profiles_cfg.pop("target_year")

# Create data model instances 
grid = GridCaseModel(**grid_cfg)
profiles = ShortTermUncertaintyProfile(**profiles_cfg)


# %% Load the pandapower network
net0 = pp.from_pickle(grid.pp_file)
# %% Read egid -> node mapping
mapping_file = Path(grid.egid_id_mapping_file)
map_df = pd.read_csv(mapping_file)

egid_col = "egid"
load_idx_col = "index"

mapping_load = map_df[[egid_col, load_idx_col]].copy()
mapping_load = mapping_load.dropna(subset=[egid_col, load_idx_col])

mapping_load[egid_col] = mapping_load[egid_col].astype(int)
mapping_load[load_idx_col] = mapping_load[load_idx_col].astype(int)

# %% egid -> load_idx -> bus mapping
load_to_bus = (
    net0.load.reset_index()[["index", "bus"]]
    .rename(columns={"index": "load_idx"})
    .copy()
)

load_to_bus["load_idx"] = load_to_bus["load_idx"].astype(int)
load_to_bus["bus"] = load_to_bus["bus"].astype(int)

mapping_bus = (
    mapping_load.rename(columns={load_idx_col: "load_idx"})
    .merge(load_to_bus, on="load_idx", how="inner")
)

# %% Reinforcement planning 
results = []
# Grid network 
net_plan = copy.deepcopy(net0)
profile_dir = Path(profiles.load_profiles[0])

for year in stage_years:
    parquet_file = profile_dir / f"{profiles.scenario_name.value}_{year}.parquet"
    print(f"Processing year {year}")
    df = pl.read_parquet(parquet_file)
    time_cols = [c for c in df.columns if c != "egid"]

    # Main loop over timestamps
    for tcol in time_cols:
        scen_df = build_snapshot_from_wide_profile(
            profile_df=df,
            tcol=tcol,
            mapping_bus=mapping_bus,
            cosphi=grid.cosφ,
            s_base=grid.s_base,
        )

        net_case = apply_profile_scenario_to_pandapower(net_plan, scen_df, grid.s_base)

        # initial capacities before reinforcement
        line_max_i_init = (
            net_case.line["max_i_ka"].copy() if len(net_case.line) > 0 else pd.Series(dtype=float)
        )
        trafo_sn_init = (
            net_case.trafo["sn_mva"].copy() if len(net_case.trafo) > 0 else pd.Series(dtype=float)
        )

        n_lines_total = len(net_case.line)
        n_trafos_total = len(net_case.trafo)

        # PF before reinforcement
        pp.runpp(net_case)

        cong_lines = check_line_loading(net_case, limit_percent=LIMIT)
        cong_trafos = check_trafo_loading(net_case, limit_percent=LIMIT)

        cong_lines_before = cong_lines.copy()
        cong_trafos_before = cong_trafos.copy()

        cong_rate_lines_before = len(cong_lines_before) / n_lines_total if n_lines_total > 0 else 0.0
        cong_rate_trafos_before = len(cong_trafos_before) / n_trafos_total if n_trafos_total > 0 else 0.0

        print(f"\nYear {year} | Timestamp {tcol}")
        print(f"Congestion threshold = {LIMIT:.1f}%")
    
    
        if len(cong_lines_before) > 0:
            print("\nTop congested lines before reinforcement:")
            print(cong_lines_before[["line_idx", "loading_percent"]].head(10))

        if len(cong_trafos_before) > 0:
            print("\nTop congested trafos before reinforcement:")
            print(cong_trafos_before[["trafo_idx", "loading_percent"]].head(10))
    
        reinforced_lines_all = set()
        reinforced_trafos_all = set()

        rounds_used = 0

        # Reinforce until congestion becomes zero
        while len(cong_lines) > 0 or len(cong_trafos) > 0:
            rounds_used += 1

            for lid in cong_lines["line_idx"].tolist():
                lid = int(lid)
                reinforce_line_one_step(net_case, lid, step_percent=STEP)
                reinforced_lines_all.add(lid)

            for tid in cong_trafos["trafo_idx"].tolist():
                tid = int(tid)
                reinforce_trafo_one_step(net_case, tid, step_percent=STEP)
                reinforced_trafos_all.add(tid)

            pp.runpp(net_case)

            cong_lines = check_line_loading(net_case, limit_percent=LIMIT)
            cong_trafos = check_trafo_loading(net_case, limit_percent=LIMIT)

            if rounds_used >= MAX_ROUNDS:
                print(f"Stop reached at={tcol}, of year={year}")
                break

        # Final state at this timestamp
        final_n_cong_lines = len(cong_lines)
        final_n_cong_trafos = len(cong_trafos)

        final_cong_rate_lines = final_n_cong_lines / n_lines_total if n_lines_total > 0 else 0.0
        final_cong_rate_trafos = final_n_cong_trafos / n_trafos_total if n_trafos_total > 0 else 0.0

        if len(net_case.line) > 0:
            delta_i_ka = (net_case.line["max_i_ka"] - line_max_i_init).fillna(0.0)
            lengths_km = (net_case.line["length_km"] if "length_km" in net_case.line.columns
                    else pd.Series(1.0, index=net_case.line.index)).fillna(1.0)
            from_bus_v_kv = net_case.line["from_bus"].map(net_case.bus["vn_kv"]).fillna(0.0)
            delta_line_kva = np.sqrt(3.0) * from_bus_v_kv * delta_i_ka * 1000.0
            delta_line_kw = delta_line_kva * grid.cosφ
            cost_lines = float((delta_line_kw * lengths_km * LINE_COST_PER_KM_KW).sum())
        else:
            cost_lines = 0.0

        if len(net_case.trafo) > 0:
            delta_sn_mva = (net_case.trafo["sn_mva"] - trafo_sn_init).fillna(0.0)
            delta_trafo_kva = delta_sn_mva * 1000.0
            delta_trafo_kw = delta_trafo_kva * grid.cosφ
            cost_trafos = float((delta_trafo_kw * TRAFO_COST_PER_KW).sum())
        else:
            cost_trafos = 0.0

        cost_total = cost_lines + cost_trafos


        results.append({
            "year": year,
            "time_col": tcol,
            "rounds_used": rounds_used,
            "n_congested_lines_before": len(cong_lines_before),
            "n_congested_trafos_before": len(cong_trafos_before),
            "cong_rate_lines_before": cong_rate_lines_before,
            "cong_rate_trafos_before": cong_rate_trafos_before,
            "final_n_congested_lines": final_n_cong_lines,
            "final_n_congested_trafos": final_n_cong_trafos,
            "final_cong_rate_lines": final_cong_rate_lines,
            "final_cong_rate_trafos": final_cong_rate_trafos,
            "reinforced_lines": ",".join(map(str, sorted(reinforced_lines_all))),
            "reinforced_trafos": ",".join(map(str, sorted(reinforced_trafos_all))), 
            "cost_lines": cost_lines,
            "cost_trafos": cost_trafos,
            "cost_total": cost_total,   
        })

        print(
            f"{year} | Time {tcol} | :"
            f"before_lines={len(cong_lines_before)}, "
            f"before_trafos={len(cong_trafos_before)}, "
            f"rounds={rounds_used}, "
            f"final_lines={final_n_cong_lines}, "
            f"final_trafos={final_n_cong_trafos}"
            f"cost_total={cost_total:,.2f}"
        )
        # use the new reinforced capacities as the start for the next time
        net_plan = copy.deepcopy(net_case)

# %% Final summary
summary_df = pd.DataFrame(results)
display(summary_df) # type: ignore
total_cost_chf = summary_df["cost_total"].sum()
total_cost_mchf = total_cost_chf / 1e6

print(f"Total cost [CHF]: {total_cost_chf:,.2f}")
print(f"Total cost [MCHF]: {total_cost_mchf:.6f}")
