# %%
import os

os.chdir(os.getcwd().replace("/src", ""))
# %%
from experiments import *

# %% set parameters

net = pp.from_pickle("data/simple_grid.p")
grid_data = pandapower_to_dig_a_plan_schema_with_scenarios(
    net,
    number_of_random_scenarios=10,
    p_bounds=(-0.6, 1.5),
    q_bounds=(-0.1, 0.1),
    v_bounds=(-0.1, 0.1),
    v_min=0.95,
    v_max=1.05,
)
groups = {
    0: [19, 20, 21, 29, 32, 35],
    1: [35, 30, 33, 25, 26, 27],
    2: [27, 32, 22, 23, 34],
    3: [31, 24, 28, 21, 22, 23],
    4: [34, 26, 25, 24, 31],
}

# %%
expansion_algorithm = ExpansionAlgorithm(
    grid_data=grid_data,
    each_task_memory=4 * 1024 * 1024 * 1024,  # 4 GB
    time_now=datetime.now().strftime("%Y%m%d_%H%M%S"),
    cache_dir=Path(".cache"),
    admm_groups=groups,
)

# %%
expansion_algorithm.run_pipeline()
