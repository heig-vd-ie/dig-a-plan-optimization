# %%
from logging import config
import polars as pl
import os
import pandapower as pp

from data_exporter.pandapower_to_dig_a_plan import pandapower_to_dig_a_plan_schema
from pipelines import DigAPlan
from pipelines.configs import ADMMConfig, PipelineType
from pipelines.model_managers.admm import PipelineModelManagerADMM


os.chdir(os.getcwd().replace("/src", ""))
os.environ["GRB_LICENSE_FILE"] = os.environ["HOME"] + "/gurobi_license/gurobi.lic"

# %% set parameters
LOAD_FACTOR = 1
TEST_CONFIG = [
    {"line_list": [], "switch_list": []},
    {"line_list": [6, 9], "switch_list": [25, 28]},
    {"line_list": [2, 6, 9], "switch_list": [21, 25, 28]},
    {"line_list": [16], "switch_list": [35]},
    {"line_list": [1], "switch_list": [20]},
    {"line_list": [10], "switch_list": [29]},
    {"line_list": [7, 11], "switch_list": [26, 30]},
]
NB_TEST = 0

net = pp.from_pickle("data/simple_grid.p")

net["load"]["p_mw"] = net["load"]["p_mw"] * LOAD_FACTOR
net["load"]["q_mvar"] = net["load"]["q_mvar"] * LOAD_FACTOR

net["line"].loc[:, "max_i_ka"] = 1
net["line"].loc[TEST_CONFIG[NB_TEST]["line_list"], "max_i_ka"] = 1e-2

# Optional tweaks
LOAD_FACTOR = 1.0
net["load"]["p_mw"] *= LOAD_FACTOR
net["load"]["q_mvar"] *= LOAD_FACTOR
net["line"].loc[:, "max_i_ka"] = 1.0

# %% Convert pandapower -> DigAPlan schema with a few scenarios
grid_data = pandapower_to_dig_a_plan_schema(net)


# %% Configure ADMM pipeline
config = ADMMConfig(
    verbose=False,
    pipeline_type=PipelineType.ADMM,
    # solver & model scaling
    solver_name="gurobi",
    solver_non_convex=2,
    big_m=1e3,
    small_m=1e-4,
    ρ=2.0,  # initial rho
    weight_infeasibility=1.0,
    weight_penalty=1e-6,
    weight_admm_penalty=1.0,
)

dap = DigAPlan(config=config)

# %% Build per-scenario models (instantiated inside add_grid_data)
dap.add_grid_data(grid_data)

# Sanity on manager type
assert isinstance(
    dap.model_manager, PipelineModelManagerADMM
), "PipelineType must be ADMM."

# %% Inspect sets (switch ids & scenarios)
switch_ids = dap.data_manager.edge_data.filter(pl.col("type") == "switch")[
    "edge_id"
].to_list()
scen_ids = list(grid_data.load_data.keys())
print("Switch IDs:", switch_ids)
print("Scenario IDs:", scen_ids)

# %% Run ADMM
dap.model_manager.solve_model(
    ρ=2.0,
    max_iters=50,
    eps_primal=1e-4,
    eps_dual=1e-4,
    adapt_ρ=True,
    mu=10.0,
    tau_incr=2.0,
    tau_decr=2.0,
)

# %% Inspect consensus and per-scenario deltas
print("\n--- ADMM consensus z per switch ---")
print(dap.model_manager.admm_z)  # {switch_id: z}

print("\n--- ADMM last-iterate delta (per scenario, per switch) ---")
print(dap.model_manager.δ_variable)  # Polars DF: ["SCEN","S","δ_variable"]


# %% Consensus switch states (one value per switch)
# z in [0,1]; threshold to get open/closed
switches = dap.data_manager.edge_data.filter(pl.col("type") == "switch").select(
    "eq_fk", "edge_id", "normal_open"
)

z_df = pl.DataFrame(
    {
        "edge_id": list(dap.model_manager.admm_z.keys()),
        "z": list(dap.model_manager.admm_z.values()),
    }
)

consensus_states = (
    switches.join(z_df, on="edge_id", how="inner")
    .with_columns(
        (pl.col("z") > 0.5).alias("closed"), (~(pl.col("z") > 0.5)).alias("open")
    )
    .select("eq_fk", "edge_id", "z", "normal_open", "closed", "open")
    .sort("edge_id")
)

print("\n=== ADMM consensus switch states (z) ===")
print(consensus_states)
