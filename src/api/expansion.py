from data_model.expansion import ExpansionInput, ExpansionOutput, InputObject
from api.grid_cases import get_grid_case
from experiments import *
from datetime import datetime

from data_model.sddp import BenderCuts

from helpers.json import load_obj_from_json, save_obj_to_json


def get_session_name() -> str:
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


def run_expansion(
    requests: ExpansionInput, with_ray: bool, cut_file: None | str = None
) -> ExpansionOutput:
    session_name = get_session_name()
    time_now = session_name
    (Path(".cache/algorithm") / time_now).mkdir(parents=True, exist_ok=True)
    save_obj_to_json(
        InputObject(expansion=requests, time_now=time_now, with_ray=with_ray),
        Path(".cache/algorithm") / time_now / "input.json",
    )
    _, grid_data = get_grid_case(
        requests.grid, seed=requests.seed, stu=requests.short_term_uncertainty
    )
    expansion_algorithm = ExpansionAlgorithm(
        grid_data=grid_data,
        admm_config=requests.admm_config,
        sddp_config=requests.sddp_config,
        long_term_uncertainty=requests.long_term_uncertainty,
        cache_dir=Path(".cache"),
        bender_cuts=(
            None
            if cut_file is None
            else BenderCuts(**load_obj_from_json(Path(cut_file)))
        ),
        time_now=time_now,
        each_task_memory=requests.each_task_memory,
        iterations=requests.iterations,
        seed_number=requests.seed,
        γ_cuts=1.0,
        s_base=requests.grid.s_base,
        with_ray=with_ray,
    )
    result = expansion_algorithm.run_pipeline()
    return ExpansionOutput(sddp_response=result)
