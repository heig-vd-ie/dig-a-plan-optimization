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
from experiments.reinforcement_power_flow.scenario_pp import apply_profile_scenario_to_pandapower
from experiments.reinforcement_power_flow.congestion_helpers import (
    check_line_loading, check_trafo_loading,
)
from pandapower.auxiliary import LoadflowNotConverged
import matplotlib.pyplot as plt

def remove_upper_tail(data, upper_pct=99):
    s = pd.Series(data).dropna()
    upper = np.percentile(s, upper_pct)
    return s[s <= upper].tolist()

# %% input parameters
LIMIT = 90.0
VMIN = 0.95
VMAX= 1.05

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


pv_egid_to_sgen = pl.DataFrame(pv_egid_to_sgen_rows).with_columns([
    pl.col("egid").cast(pl.Int64, strict=False),
    pl.col("sgen_idx").cast(pl.Int64, strict=False),
]).drop_nulls(["egid", "sgen_idx"])


line_loading_count_year = []
trafo_loading_count_year = []

bus_voltage_dist_year = []

line_loading_dist_year = []
trafo_loading_dist_year = []




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
    
    year_line_counts = []
    year_trafo_counts = []
    
    year_bus_voltage_dist = []
    
    year_line_loading_dist = []
    year_trafo_loading_dist = []
    
    for tcol in time_cols:
    
        net_case = apply_profile_scenario_to_pandapower(
            net0=net0,
            load_data=load_data,
            pv_data=pv_data,
            tcol=tcol,
            mapping_load=mapping_load,
            pv_egid_to_sgen=pv_egid_to_sgen,
            cosphi=grid.cosφ,
        )
        
        total_p_load_mw = float(net_case.load["p_mw"].sum())
        total_q_load_mvar = float(net_case.load["q_mvar"].sum())
        total_pv_mw = float(net_case.sgen["p_mw"].sum())

        
        pp.runpp(
            net_case, 
            algorithm="nr", 
            max_iteration=50, 
            tolerance_mva=1e-6, 
            init ="auto",
            check_connectivity=True,
        )
            
        line_loading = net_case.res_line["loading_percent"].dropna()
        trafo_loading = net_case.res_trafo["loading_percent"].dropna()
            
        vm = net_case.res_bus["vm_pu"].dropna()
        
        year_bus_voltage_dist.extend(vm.tolist())
            
        year_line_counts.append(int((line_loading > LIMIT).sum()))
        year_trafo_counts.append(int((trafo_loading > LIMIT).sum()))

        
        year_line_loading_dist.extend(line_loading.tolist())
        year_trafo_loading_dist.extend(trafo_loading.tolist())

   
            
    
    line_loading_count_year.append(year_line_counts)
    trafo_loading_count_year.append(year_trafo_counts)
    
    bus_voltage_dist_year.append(year_bus_voltage_dist)
    
    
    line_loading_dist_year.append(year_line_loading_dist)
    trafo_loading_dist_year.append(year_trafo_loading_dist)
    
    
line_loading_dist_year_filtered = [
    remove_upper_tail(x, upper_pct=99) for x in line_loading_dist_year
]


trafo_loading_dist_year_filtered = [
    remove_upper_tail(x, upper_pct=99) for x in trafo_loading_dist_year
]

bus_voltage_dist_year_filtered = [
    remove_upper_tail(x, upper_pct=99) for x in bus_voltage_dist_year
]

line_loading_count_year_filtered = [
    remove_upper_tail(x, upper_pct=99) for x in line_loading_count_year
]

trafo_loading_count_year_filtered = [
    remove_upper_tail(x, upper_pct=99) for x in trafo_loading_count_year
]

line_plot_data = [pd.Series(x).dropna().tolist() for x in line_loading_count_year_filtered]
trafo_plot_data = [pd.Series(x).dropna().tolist() for x in trafo_loading_count_year_filtered]
bus_voltage_plot_data = [pd.Series(x).dropna().tolist() for x in bus_voltage_dist_year]
line_loading_plot_data = [pd.Series(x).dropna().tolist() for x in line_loading_dist_year_filtered]
trafo_loading_plot_data = [pd.Series(x).dropna().tolist() for x in trafo_loading_dist_year_filtered]

# %% boxplot for lines
plt.figure(figsize=(10, 5))
plt.boxplot(line_plot_data, labels=stage_years)
plt.xlabel("Year")
plt.ylabel(f"Number of lines with loading > {LIMIT:.0f}%")
plt.grid(True)
plt.tight_layout()
plt.show()

# boxplot for trafos
plt.figure(figsize=(10, 5))
plt.boxplot(trafo_plot_data, labels=stage_years)
plt.xlabel("Year")
plt.ylabel(f"Number of trafos with loading > {LIMIT:.0f}%")
plt.grid(True)
plt.tight_layout()
plt.show()

# %% Boxplot: voltage distribution


plt.figure(figsize=(10, 5))
plt.boxplot(bus_voltage_plot_data, labels=stage_years)
plt.axhline(VMIN, linestyle="--", linewidth=1, label=f"VMIN={VMIN}")
plt.axhline(VMAX, linestyle="--", linewidth=1, label=f"VMAX={VMAX}")
plt.xlabel("Year")
plt.ylabel("Bus voltage [pu]")
plt.title("Before reinforcement: bus voltage distribution")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# %% line loading percent distribution
plt.figure(figsize=(10, 5))
plt.boxplot(line_loading_plot_data, labels=stage_years)
plt.axhline(LIMIT, linestyle="--", linewidth=1, label=f"Limit={LIMIT:.0f}%")
plt.xlabel("Year")
plt.ylabel("Line loading percent")
plt.title("Distribution of line loading percent")
plt.grid(True)
plt.tight_layout()
plt.show()

# %% trafo loading percent distribution
plt.figure(figsize=(10, 5))
plt.boxplot(trafo_loading_plot_data, labels=stage_years)
plt.axhline(LIMIT, linestyle="--", linewidth=1, label=f"Limit={LIMIT:.0f}%")
plt.xlabel("Year")
plt.ylabel("Transformer loading percent")
plt.title("Distribution of transformer loading percent")
plt.grid(True)
plt.tight_layout()
plt.show()

