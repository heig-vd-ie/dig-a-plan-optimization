# %%
from experiments import *
from pathlib import Path

# %% set parameters
PROJECT_ROOT = Path(__file__).resolve().parents[2]

seed = 42
pp_path = PROJECT_ROOT / ".cache" / "input" / "boisy" / "boisy_grid.p"

grid = GridCaseModel(
    pp_file=str(pp_path),
    s_base=1e6,
)
stu = ShortTermUncertaintyRandom(
    n_scenarios=10,
    v_bounds=(-0.07, 0.07),
    p_bounds=(-0.5, 1.0),
    q_bounds=(-0.5, 0.5),
)

# %% --- CLEAN NULLS IN THE RAW PANDAPOWER NET (same as your 2nd script) ---
net, grid_data = get_grid_case(grid=grid, seed=seed, stu=stu)

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
    time_now=datetime.now().strftime("%Y%m%d_%H%M%S"),
    cache_dir=Path(".cache"),
    admm_config=ADMMConfig(
        groups=10,  # TODO: set number of groups for actual boisy grid to 40
        # time_limit=10,  # TODO: set time limit to 10 seconds for actual boisy grid
        # solver_non_convex=2,  # Set non-convex parameters to 2 for Boisy grid
    ),
    sddp_config=SDDPConfig(),
    long_term_uncertainty=LongTermUncertainty(),
)

# %%
expansion_algorithm.run_pipeline()
