from pathlib import Path
from datetime import datetime
from api.grid_cases import get_grid_case_for_expansion
from data_model.expansion import ExpansionInput, ExpansionOutput
from data_model.sddp import BenderCuts
from pipeline_expansion.algorithm import ExpansionAlgorithm
from helpers.json import load_obj_from_json, save_obj_to_json
from konfig import settings

INPUT_FILENAME = "input.json"


def get_session_name() -> str:
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


def run_expansion(
    requests: ExpansionInput,
    with_ray: bool,
    cut_file: None | str = None,
    time_now: str | None = None,
) -> ExpansionOutput:
    if not time_now:
        time_now = get_session_name()
    (Path(settings.cache.outputs_expansion) / time_now).mkdir(
        parents=True, exist_ok=True
    )
    save_obj_to_json(
        requests,
        Path(settings.cache.outputs_expansion) / time_now / INPUT_FILENAME,
    )
    _, grid_data, load_potential, pv_potential = get_grid_case_for_expansion(
        requests.grid,
        seed=requests.seed,
        stu=requests.short_term_uncertainty,
        profiles=requests.profiles,
    )

    bender_cuts = (
        None if cut_file is None else BenderCuts(**load_obj_from_json(Path(cut_file)))
    )
    expansion_algorithm = ExpansionAlgorithm(
        grid_data=grid_data,
        admm_config=requests.admm_config,
        sddp_config=requests.sddp_config,
        bender_cuts=bender_cuts,
        load_potential=load_potential,
        each_task_memory=requests.each_task_memory,
        fixed_switches=requests.admm_config.fixed_switches,
        pv_potential=pv_potential,
        cache_dir=Path(settings.cache.outputs_expansion),
        time_now=time_now,
        iterations=requests.iterations,
        seed_number=requests.seed,
        γ_cuts=1.0,
        s_base=requests.grid.s_base,
        with_ray=with_ray,
    )
    result = expansion_algorithm.run_pipeline()
    return ExpansionOutput(sddp_response=result)
