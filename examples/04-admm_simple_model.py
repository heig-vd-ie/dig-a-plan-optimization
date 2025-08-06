# %%
from examples import *

# %% set parameters

net = pp.from_pickle("data/simple_grid.p")
grid_data = pandapower_to_dig_a_plan_schema(net)
groups = {
    0: [19, 20, 21, 29, 32, 35],
    1: [35, 30, 33, 25, 26, 27],
    2: [27, 32, 22, 23, 34],
    3: [31, 24, 28, 21, 22, 23],
    4: [34, 26, 25, 24, 31],
}


# %% Configure ADMM pipeline
config = ADMMConfig(
    verbose=False,
    pipeline_type=PipelineType.ADMM,
    solver_name="gurobi",
    solver_non_convex=2,
    big_m=1e3,
    ε=1,
    ρ=2.0,
    γ_infeasibility=100.0,
    γ_admm_penalty=1.0,
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
    max_iters=10,
    μ=10.0,
    τ_incr=2.0,
    τ_decr=2.0,
    groups=groups,
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

# %% Extract and compare results
# %% compare DigAPlan results with pandapower results
node_data, edge_data = compare_dig_a_plan_with_pandapower(dig_a_plan=dap, net=net)
# %% plot the grid annotated with DigAPlan results
fig = plot_grid_from_pandapower(
    net,
    dap,
    switch_status=pl_to_dict(consensus_states.select("eq_fk", "closed")),
)
