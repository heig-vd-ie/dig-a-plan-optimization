from api import *
from typing import Dict, List

from pipelines.expansion.models.request import RiskMeasureType
from pipelines.expansion.models.response import ExpansionResponse


class ExpansionInput(GridCaseModel):
    groups: int | Dict[int, List[int]] = 10
    admm_iterations: int = 10
    n_admm_simulations: int = 10
    admm_time_limit: int = 10
    solver_non_convex: bool = True
    n_stages: int = 3
    initial_budget: float = 1e6
    discount_rate: float = 0.05
    years_per_stage: int = 1
    sddp_iteration_limit: int = 10
    sddp_simulations: int = 100
    risk_measure_type: RiskMeasureType = RiskMeasureType.EXPECTATION
    risk_measure_param: float = 0.1
    δ_load_var: float = 0.1
    δ_pv_var: float = 0.1
    δ_b_var: float = 0.1
    number_of_sddp_scenarios: int = 100
    investment_costs: float = 1e3
    penalty_costs_load: float = 1e3
    penalty_costs_pv: float = 1e3


class ExpansionOutput(ExpansionResponse):
    pass


def run_expansion(input: ExpansionInput, with_ray: bool) -> ExpansionOutput:
    _, grid_data = get_grid_case(input)
    expansion_algorithm = ExpansionAlgorithm(
        grid_data=grid_data,
        cache_dir=Path(".cache"),
        admm_groups=input.groups,
        iterations=input.admm_iterations,
        n_admm_simulations=input.n_admm_simulations,
        seed_number=input.seed,
        time_limit=input.admm_time_limit,
        solver_non_convex=2 if input.solver_non_convex else 0,
        n_stages=input.n_stages,
        initial_budget=input.initial_budget,
        discount_rate=input.discount_rate,
        years_per_stage=input.years_per_stage,
        γ_cuts=1.0,
        sddp_iteration_limit=input.sddp_iteration_limit,
        sddp_simulations=input.sddp_simulations,
        risk_measure_type=input.risk_measure_type,
        risk_measure_param=input.risk_measure_param,
        δ_load_var=input.δ_load_var,
        δ_pv_var=input.δ_pv_var,
        δ_b_var=input.δ_b_var,
        number_of_sddp_scenarios=input.number_of_sddp_scenarios,
        investment_costs=input.investment_costs,
        penalty_costs_load=input.penalty_costs_load,
        penalty_costs_pv=input.penalty_costs_pv,
        with_ray=with_ray,
    )
    result = expansion_algorithm.run_pipeline()
    return ExpansionOutput(**result.model_dump())
