from data_model.expansion import ExpansionInput, ExpansionOutput, InputObject
from api.grid_cases import get_grid_case
from experiments import *
from datetime import datetime

from data_model.sddp import BenderCuts

from helpers.json import load_obj_from_json, save_obj_to_json


def get_session_name() -> str:
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


def run_expansion(
    requets: ExpansionInput, with_ray: bool, cut_file: None | str = None
) -> ExpansionOutput:
    session_name = get_session_name()
    time_now = session_name
    (Path(".cache/algorithm") / time_now).mkdir(parents=True, exist_ok=True)
    save_obj_to_json(
        InputObject(expansion=requets, time_now=time_now, with_ray=with_ray),
        Path(".cache/algorithm") / time_now / "input.json",
    )
    _, grid_data = get_grid_case(
        requets.grid, seed=requets.seed, stu=requets.short_term_uncertainty
    )
    expansion_algorithm = ExpansionAlgorithm(
        grid_data=grid_data,
        admm_config=requets.admm_config,
        cache_dir=Path(".cache"),
        bender_cuts=(
            None
            if cut_file is None
            else BenderCuts(**load_obj_from_json(Path(cut_file)))
        ),
        time_now=time_now,
        each_task_memory=requets.each_task_memory,
        iterations=requets.iterations,
        n_admm_simulations=requets.sddp_config.n_simulations,
        seed_number=requets.seed,
        n_stages=requets.long_term_uncertainty.n_stages,
        initial_budget=requets.sddp_params.initial_budget,
        discount_rate=requets.sddp_params.discount_rate,
        years_per_stage=requets.sddp_params.years_per_stage,
        γ_cuts=1.0,
        sddp_iteration_limit=requets.sddp_config.iterations,
        sddp_simulations=requets.sddp_config.n_simulations,
        risk_measure_type=requets.sddp_params.risk_measure_type,
        risk_measure_param=requets.sddp_params.risk_measure_param,
        δ_load_var=requets.long_term_uncertainty.δ_load_var,
        δ_pv_var=requets.long_term_uncertainty.δ_pv_var,
        δ_b_var=requets.long_term_uncertainty.δ_b_var,
        number_of_sddp_scenarios=requets.long_term_uncertainty.number_of_scenarios,
        expansion_line_cost_per_km_kw=requets.sddp_params.expansion_line_cost_per_km_kw,
        expansion_transformer_cost_per_kw=requets.sddp_params.expansion_transformer_cost_per_kw,
        penalty_cost_per_consumption_kw=requets.sddp_params.penalty_cost_per_consumption_kw,
        penalty_cost_per_production_kw=requets.sddp_params.penalty_cost_per_production_kw,
        s_base=requets.grid.s_base,
        with_ray=with_ray,
    )
    result = expansion_algorithm.run_pipeline()
    return ExpansionOutput(sddp_response=result)
