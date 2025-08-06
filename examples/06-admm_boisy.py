# %%
import os

os.chdir(os.getcwd().replace("/src", ""))

# %%
from examples import *

# %% Convert pandapower -> DigAPlan schema with a few scenarios
if USE_SIMPLIFIED_GRID := True:
    net = pp.from_pickle(".cache/boisy_grid_simplified.p")
    grid_data = pandapower_to_dig_a_plan_schema(
        net,
        number_of_random_scenarios=10,
    )
else:
    net = pp.from_pickle(".cache/boisy_grid.p")
    grid_data = pandapower_to_dig_a_plan_schema(
        net,
        number_of_random_scenarios=30,
    )

# %% convert pandapower grid to DigAPlan grid data

grid_data.edge_data = grid_data.edge_data.with_columns(
    pl.when(c(col) < 1e-3).then(pl.lit(0)).otherwise(c(col)).alias(col)
    for col in ["b_pu", "r_pu", "x_pu"]
).with_columns(
    c("normal_open").fill_null(False),
)

# %% Configure ADMM pipeline
config = ADMMConfig(
    verbose=False,
    pipeline_type=PipelineType.ADMM,
    solver_name="gurobi",
    solver_non_convex=0,
    big_m=1e3,
    ε=1e-4,
    ρ=2.0,
    γ_infeasibility=100.0,
    γ_admm_penalty=1.0,
    time_limit=1,
    max_iters=10,
    μ=10.0,
    τ_incr=2.0,
    τ_decr=2.0,
    mutation_factor=2,
    groups=10,
)

dap = DigAPlanADMM(config=config)

# %% Build per-scenario models (instantiated inside add_grid_data)
dap.add_grid_data(grid_data)


# %% Run ADMM
dap.model_manager.solve_model()

# %% Inspect consensus and per-scenario deltas

print("\n=== ADMM consensus switch states (z) ===")
print(dap.model_manager.z_variable)
