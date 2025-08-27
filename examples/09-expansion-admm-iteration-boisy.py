# %%
import os

os.chdir(os.getcwd().replace("/src", ""))
# %%
from examples import *

# %% set parameters

if USE_SIMPLIFIED_GRID := True:
    net = pp.from_pickle(".cache/boisy_grid_simplified.p")
    grid_data = pandapower_to_dig_a_plan_schema(
        net,
        number_of_random_scenarios=10,
        v_bounds=(-0.07, 0.07),
        p_bounds=(-0.5, 1.0),
        q_bounds=(-0.5, 0.5),
        taps=[95, 98, 99, 100, 101, 102, 105],
    )
else:
    net = pp.from_pickle(".cache/boisy_grid.p")
    grid_data = pandapower_to_dig_a_plan_schema(
        net,
        number_of_random_scenarios=10,
        taps=[95, 98, 99, 100, 101, 102, 105],
    )

grid_data.edge_data = grid_data.edge_data.with_columns(
    pl.when(c(col) < 1e-3).then(pl.lit(0)).otherwise(c(col)).alias(col)
    for col in ["b_pu", "r_pu", "x_pu"]
).with_columns(
    c("normal_open").fill_null(False),
)
# %%
expansion_algorithm = ExpansionAlgorithm(
    grid_data=grid_data,
    each_task_memory=4 * 1024 * 1024 * 1024,  # 4 GB
    cache_dir=Path(".cache"),
    admm_groups=10,  # TODO: set number of groups for actual boisy grid to 40
    # time_limit=10,  # TODO: set time limit to 10 seconds for actual boisy grid
    # solver_non_convex=2,  # Set non-convex parameters to 2 for Boisy grid
)

# %%
expansion_algorithm.run_pipeline()
