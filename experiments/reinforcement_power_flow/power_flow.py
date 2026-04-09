# %% Import libraries
from pathlib import Path
import json
import copy
import pickle

import numpy as np
import pandas as pd
import pandapower as pp
import polars as pl
import matplotlib.pyplot as plt

from data_model import GridCaseModel, ShortTermUncertaintyProfile
from data_model.kace import DiscreteScenario
from experiments.reinforcement_power_flow.scenario_pp import apply_profile_scenario_to_pandapower
from experiments.reinforcement_power_flow.congestion_helpers import (
    check_line_loading, check_trafo_loading,
)

# %% input parameters for reinforcement and congestion settings
LIMIT = 90.0
VMIN = 0.95
VMAX = 1.05
MAX_ROUNDS = 50  
LINE_COST_PER_KM_KW = 1752
TRAFO_COST_PER_KW = 1314
DISCOUNT_RATE = 0.05


# %% Load JSON config 
project_root = Path.cwd().parent
json_file = project_root / "experiments" / "reinforcement_power_flow" / "00-power-flow.json"

results_dir = project_root / ".cache" / "results"
power_flow_results_file = project_root / ".cache" / "power_flow_results_after_reinforcement.pkl"
summary_results_file = project_root / ".cache" / "reinforcement_summary.pkl"

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
egid_col = "egid"
load_idx_col = "index"

mapping_load = (
    pl.read_csv(mapping_file)
    .select([
        pl.col(egid_col).alias("egid"),
        pl.col(load_idx_col).alias("load_idx"),
    ])
    .with_columns([
        pl.col("egid").cast(pl.Int64, strict=False),
        pl.col("load_idx").cast(pl.Int64, strict=False),
    ])
    .drop_nulls(["egid", "load_idx"])
)

load_idx_to_bus = (
    pl.DataFrame({
        "load_idx": net0.load.index.to_list(),
        "bus": net0.load["bus"].to_list(),
    })
    .with_columns([
        pl.col("load_idx").cast(pl.Int64, strict=False),
        pl.col("bus").cast(pl.Int64, strict=False),
    ])
)

mapping_pv = (
    mapping_load
    .join(load_idx_to_bus, on="load_idx", how="left")
    .drop_nulls(["egid", "load_idx", "bus"])
    .unique(subset=["egid"], keep="first")
)

# %% Add PV generators
net0 = copy.deepcopy(net0)

pv_egid_to_sgen_rows = []

for row in mapping_pv.select(["egid", "bus"]).iter_rows(named=True):
    egid = int(row["egid"])
    bus_idx = int(row["bus"])

    sgen_idx = pp.create_sgen(
        net0,
        bus=bus_idx,
        p_mw=0.0,
        q_mvar=0.0,
        name=f"pv_sgen_egid_{egid}",
        type="PV",
        in_service=True,
    )

    pv_egid_to_sgen_rows.append({
        "egid": egid,
        "sgen_idx": int(sgen_idx),
    })


pv_egid_to_sgen = (
    pl.DataFrame(pv_egid_to_sgen_rows)
    .with_columns([
        pl.col("egid").cast(pl.Int64, strict=False),
        pl.col("sgen_idx").cast(pl.Int64, strict=False),
    ]).drop_nulls(["egid", "sgen_idx"])
)

line_max_i_base = net0.line["max_i_ka"].copy()
trafo_sn_base = net0.trafo["sn_mva"].copy()

# %% Reinforcement planning 
results = []
yearly_results = []

# after-reinforcement distributions
line_loading_count_year = []
trafo_loading_count_year = []
bus_voltage_dist_year = []
line_loading_dist_year = []
trafo_loading_dist_year = []


# Grid network 
net_plan = copy.deepcopy(net0)
base_year = min(stage_years)
line_length_km = pd.Series(1.0, index=net0.line.index, dtype=float)


for year in stage_years:
    print(f"Processing year {year}")
    
    load_data = pl.concat([
        pl.read_parquet(Path(profiles.load_profiles[0]) / f"{profiles.scenario_name.value}_{year}.parquet"),
        pl.read_parquet(Path(profiles.load_profiles[1]) / f"{profiles.scenario_name.value}_{year}.parquet"),
    ], how="vertical").with_columns(
        pl.col("egid").cast(pl.Int64, strict=False)
    )   

    pv_data = pl.read_parquet(
        Path(profiles.pv_profile) / f"{profiles.scenario_name.value}_{year}.parquet"
    ).with_columns(
        pl.col("egid").cast(pl.Int64, strict=False)
    )
    
    time_cols = [c for c in load_data.columns if c != "egid"]
    
    
    load_data = (
        load_data
        .group_by("egid")
        .agg([pl.col(c).sum().alias(c) for c in time_cols])
        .sort("egid")
    )
    
    line_max_i_init = net_plan.line["max_i_ka"].copy()
    trafo_sn_init = net_plan.trafo["sn_mva"].copy()
    
    year_line_counts = []
    year_trafo_counts = []
    year_bus_voltage_dist = []
    year_line_loading_dist = []
    year_trafo_loading_dist = []


    # Main loop over timestamps
    for tcol in time_cols:

        net_case = apply_profile_scenario_to_pandapower(
            net0=net_plan,
            load_data=load_data,
            pv_data=pv_data,
            tcol=tcol,
            mapping_load=mapping_load,
            pv_egid_to_sgen=pv_egid_to_sgen,
            cosphi=grid.cosφ,
        )
        
        # pandapower run before reinforcement
        pp.runpp(net_case)
        
        cong_lines = check_line_loading(net_case, limit_percent=LIMIT)
        cong_trafos = check_trafo_loading(net_case, limit_percent=LIMIT)

        cong_lines_before = cong_lines.copy()
        cong_trafos_before = cong_trafos.copy()

        n_lines_total = len(net_case.line)
        n_trafos_total = len(net_case.trafo)
        
    
        reinforced_lines = set()
        reinforced_trafos = set()

        rounds_used = 0
            
        while len(cong_lines) > 0 or len(cong_trafos) > 0:
            rounds_used += 1

            net_case.line["loading_percent"] = net_case.res_line["loading_percent"]
            overloaded_line_idx = net_case.line.index[
                net_case.line["loading_percent"] >= LIMIT
            ].tolist()
            reinforced_lines.update(map(int, overloaded_line_idx))
            
            net_case.line["max_i_ka"] = net_case.line.apply(
                lambda x: (
                    x["max_i_ka"]
                    if x["loading_percent"] < LIMIT
                    else x["max_i_ka"] * x["loading_percent"] * 1.2 / 100
                ),
                axis=1,
            )
            net_case.line = net_case.line.drop(columns="loading_percent")
 
            net_case.trafo["loading_percent"] = net_case.res_trafo["loading_percent"]
            
            overloaded_trafo_idx = net_case.trafo.index[
                net_case.trafo["loading_percent"] >= LIMIT
            ].tolist()
            reinforced_trafos.update(map(int, overloaded_trafo_idx))
            
            net_case.trafo["sn_mva"] = net_case.trafo.apply(
                lambda x: (
                    x["sn_mva"]
                    if x["loading_percent"] < LIMIT
                    else x["sn_mva"] * x["loading_percent"] * 1.2 / 100
                ),
                axis=1,
            )
            net_case.trafo = net_case.trafo.drop(columns="loading_percent")

            pp.runpp(net_case)

            cong_lines = check_line_loading(net_case, limit_percent=LIMIT)
            cong_trafos = check_trafo_loading(net_case, limit_percent=LIMIT)

            if rounds_used >= MAX_ROUNDS:
                print(f"Stop reached at={tcol}, of year={year}")
                break


        # Final reinforced state
        final_n_cong_lines = len(cong_lines)
        final_n_cong_trafos = len(cong_trafos)
        line_loading_final = net_case.res_line["loading_percent"].dropna()
        trafo_loading_final = net_case.res_trafo["loading_percent"].dropna()
        vm_final = net_case.res_bus["vm_pu"].dropna()

        year_line_counts.append(int((line_loading_final > LIMIT).sum()))
        year_trafo_counts.append(int((trafo_loading_final > LIMIT).sum()))
        year_bus_voltage_dist.extend(vm_final.tolist())
        year_line_loading_dist.extend(line_loading_final.tolist())
        year_trafo_loading_dist.extend(trafo_loading_final.tolist())


        results.append({
            "year": year,
            "time_col": tcol,
            "rounds_used": rounds_used,
            "n_congested_lines_before": len(cong_lines_before),
            "n_congested_trafos_before": len(cong_trafos_before),
            "cong_rate_lines_before": len(cong_lines_before) / n_lines_total if n_lines_total else 0.0,
            "cong_rate_trafos_before": len(cong_trafos_before) / n_trafos_total if n_trafos_total else 0.0,
            "final_n_congested_lines": final_n_cong_lines,
            "final_n_congested_trafos": final_n_cong_trafos,
            "final_cong_rate_lines": final_n_cong_lines / n_lines_total if n_lines_total else 0.0,
            "final_cong_rate_trafos": final_n_cong_trafos / n_trafos_total if n_trafos_total else 0.0,
            "reinforced_lines": ",".join(map(str, sorted(reinforced_lines))),
            "reinforced_trafos": ",".join(map(str, sorted(reinforced_trafos))),  
        })


        # use the new reinforced capacities as the start for the next time
        net_plan = copy.deepcopy(net_case)
        

    
    line_loading_count_year.append(year_line_counts)
    trafo_loading_count_year.append(year_trafo_counts)
    bus_voltage_dist_year.append(year_bus_voltage_dist)
    line_loading_dist_year.append(year_line_loading_dist)
    trafo_loading_dist_year.append(year_trafo_loading_dist)
        
    # yearly reinforcement costs
    delta_i_ka = net_plan.line["max_i_ka"].sub(line_max_i_init, fill_value=0.0)
    delta_sn_mva = net_plan.trafo["sn_mva"].sub(trafo_sn_init, fill_value=0.0)

    
    lengths_km = net_plan.line.get("length_km", line_length_km).fillna(1.0)
    from_bus_v_kv = net_plan.line["from_bus"].map(net_plan.bus["vn_kv"]).fillna(0.0)

    delta_line_kva = np.sqrt(3.0) * from_bus_v_kv * delta_i_ka * 1000.0
    delta_line_kw = delta_line_kva * grid.cosφ
    cost_lines = float((delta_line_kw * lengths_km * LINE_COST_PER_KM_KW).sum())

    delta_trafo_kva = delta_sn_mva * 1000.0
    delta_trafo_kw = delta_trafo_kva * grid.cosφ
    cost_trafos = float((delta_trafo_kw * TRAFO_COST_PER_KW).sum())

    cost_total_year = cost_lines + cost_trafos
    years_from_base = year - base_year
    discount_factor = (1.0 + DISCOUNT_RATE) ** years_from_base
    npv_cost_year = cost_total_year / discount_factor
    
    yearly_results.append({
        "year": year,
        "npv_cost_total": npv_cost_year,
    })
    
    print(f"Year {year} reinforcement cost: CHF {cost_total_year:,.2f} | NPV Cost: CHF {npv_cost_year:,.2f}"
    )
    
# %% Save power flow results in .cache
power_flow_data = {
    "line_loading_count_year": line_loading_count_year,
    "trafo_loading_count_year": trafo_loading_count_year,
    "bus_voltage_dist_year": bus_voltage_dist_year,
    "line_loading_dist_year": line_loading_dist_year,
    "trafo_loading_dist_year": trafo_loading_dist_year,
    "results": results,
    "yearly_results": yearly_results,
}

with open(power_flow_results_file, "wb") as f:
    pickle.dump(power_flow_data, f)

print(f"Saved power flow results file to: {power_flow_results_file}")


# %% Final summary
summary_df = pd.DataFrame(results)
yearly_df = pd.DataFrame(yearly_results)

summary_results_data = {
    "summary_df": summary_df,
    "yearly_df": yearly_df,
}

with open(summary_results_file, "wb") as f:
    pickle.dump(summary_results_data, f)
    
print(summary_df) 
print(yearly_df) 

total_npv_chf = yearly_df["npv_cost_total"].sum()
total_cost_mchf = total_npv_chf / 1e6

print(f"Total cost [MCHF]: {total_cost_mchf:.6f}")


# %% Boxplots after reinforcement
# plt.figure(figsize=(10, 5))
# plt.boxplot(line_loading_count_year, labels=stage_years)
# plt.xlabel("Year")
# plt.ylabel(f"Number of lines with loading > {LIMIT:.0f}%")
# plt.title("After reinforcement: overloaded lines count")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(10, 5))
# plt.boxplot(trafo_loading_count_year, labels=stage_years)
# plt.xlabel("Year")
# plt.ylabel(f"Number of trafos with loading > {LIMIT:.0f}%")
# plt.title("After reinforcement: overloaded trafos count")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

fig = plt.figure(figsize=(10, 5))
plt.boxplot(bus_voltage_dist_year, labels=stage_years)
plt.axhline(VMIN, linestyle="--", linewidth=1, label=f"VMIN={VMIN}")
plt.axhline(VMAX, linestyle="--", linewidth=1, label=f"VMAX={VMAX}")
plt.xlabel("Year")
plt.ylabel("Bus voltage [pu]")
plt.title("After reinforcement: bus voltage distribution")
plt.grid(True)
plt.legend()
plt.tight_layout()
fig.savefig(results_dir / "voltage_after_Basic.png", dpi=200, bbox_inches="tight")
plt.show()

fig = plt.figure(figsize=(10, 5))
plt.boxplot(line_loading_dist_year, labels=stage_years)
plt.axhline(LIMIT, linestyle="--", linewidth=1, label=f"Limit={LIMIT:.0f}%")
plt.xlabel("Year")
plt.ylabel("Line loading percent")
plt.title("After reinforcement: line loading distribution")
plt.grid(True)
plt.legend()
plt.tight_layout()
fig.savefig(results_dir / "line_distribution_loading_after_Basic.png", dpi=200, bbox_inches="tight")
plt.show()

fig = plt.figure(figsize=(10, 5))
plt.boxplot(trafo_loading_dist_year, labels=stage_years)
plt.axhline(LIMIT, linestyle="--", linewidth=1, label=f"Limit={LIMIT:.0f}%")
plt.xlabel("Year")
plt.ylabel("Transformer loading percent")
plt.title("After reinforcement: transformer loading distribution")
plt.grid(True)
plt.legend()
plt.tight_layout()
fig.savefig(results_dir / "trafo_distribution_loading_after_Basic.png", dpi=200, bbox_inches="tight")
plt.show()


# %%
