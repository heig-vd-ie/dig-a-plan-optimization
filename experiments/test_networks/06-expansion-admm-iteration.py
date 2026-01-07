# %% import libraries

from experiments import *
from pathlib import Path
from api.grid_cases import get_grid_case
from data_model.kace import GridCaseModel
from data_model.reconfiguration import ShortTermUncertainty

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# %% --- Load net via API  ---
grid = GridCaseModel(
    pp_file=str(PROJECT_ROOT / "examples" / "ieee-33" / "simple_grid.p"),
    s_base=1e6,
)
stu = ShortTermUncertainty()

net, _ = get_grid_case(grid=grid, seed=42, stu=stu)
# %% --- build grid data with scenarios ---
net.bus["max_vm_pu"] = 1.05
net.bus["min_vm_pu"] = 0.95
grid_data = pandapower_to_dig_a_plan_schema_with_scenarios(
    net,
    number_of_random_scenarios=100,
    p_bounds=(-0.6, 1.5),
    q_bounds=(-0.1, 0.1),
    v_bounds=(-0.1, 0.1),
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
