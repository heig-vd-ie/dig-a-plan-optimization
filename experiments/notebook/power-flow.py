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

# %%
