from api.models import ExpansionInput, ExpansionOutput, InputObject
from api.grid_cases import get_grid_case
from experiments import *
from datetime import datetime

from pipeline_expansion.models.request import BenderCuts

from helpers.json import load_obj_from_json, save_obj_to_json


def get_session_name() -> str:
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


def run_expansion(
    input: ExpansionInput, with_ray: bool, cut_file: None | str = None
) -> ExpansionOutput:
    session_name = get_session_name()
    time_now = session_name
    (Path(".cache/algorithm") / time_now).mkdir(parents=True, exist_ok=True)
    save_obj_to_json(
        InputObject(expansion=input, time_now=time_now, with_ray=with_ray),
        Path(".cache/algorithm") / time_now / "input.json",
    )
    _, grid_data = get_grid_case(
        input.grid, seed=input.seed, stu=input.short_term_uncertainty
    )
    expansion_algorithm = ExpansionAlgorithm(
        grid_data=grid_data,
        admm_voll=input.admm_params.voll,
        admm_volp=input.admm_params.volp,
        cache_dir=Path(".cache"),
        bender_cuts=(
            None
            if cut_file is None
            else BenderCuts(**load_obj_from_json(Path(cut_file)))
        ),
        time_now=time_now,
        each_task_memory=input.each_task_memory,
        admm_groups=input.admm_config.groups,
        iterations=input.iterations,
        n_admm_simulations=input.admm_config.n_simulations,
        seed_number=input.seed,
        time_limit=input.admm_config.time_limit,
        solver_non_convex=2 if input.admm_config.solver_non_convex else 0,
        n_stages=input.long_term_uncertainty.n_stages,
        initial_budget=input.sddp_params.initial_budget,
        discount_rate=input.sddp_params.discount_rate,
        years_per_stage=input.sddp_params.years_per_stage,
        γ_cuts=1.0,
        sddp_iteration_limit=input.sddp_config.iterations,
        sddp_simulations=input.sddp_config.n_simulations,
        risk_measure_type=input.sddp_params.risk_measure_type,
        risk_measure_param=input.sddp_params.risk_measure_param,
        δ_load_var=input.long_term_uncertainty.δ_load_var,
        δ_pv_var=input.long_term_uncertainty.δ_pv_var,
        δ_b_var=input.long_term_uncertainty.δ_b_var,
        number_of_sddp_scenarios=input.long_term_uncertainty.number_of_scenarios,
        expansion_line_cost_per_km_kw=input.sddp_params.expansion_line_cost_per_km_kw,
        expansion_transformer_cost_per_kw=input.sddp_params.expansion_transformer_cost_per_kw,
        penalty_cost_per_consumption_kw=input.sddp_params.penalty_cost_per_consumption_kw,
        penalty_cost_per_production_kw=input.sddp_params.penalty_cost_per_production_kw,
        s_base=input.grid.s_base,
        admm_max_iters=input.admm_config.iterations,
        with_ray=with_ray,
    )
    result = expansion_algorithm.run_pipeline()
    return ExpansionOutput(sddp_response=result)
