from api import *
from pydantic import Field
from typing import Dict, List
from pipelines.expansion.models.request import RiskMeasureType
from pipelines.expansion.models.response import ExpansionResponse


class GridModel(BaseModel):
    kace: GridCase = GridCase.SIMPLE_GRID
    s_base: float = 1e6
    taps: List[int] = list(range(95, 105, 1))
    v_min: float = 0.9
    v_max: float = 1.1
    groups: int | Dict[int, List[int]] = 10


class ShortTermUncertainty(BaseModel):
    number_of_scenarios: int = 10
    p_bounds: Tuple[float, float] = (-0.2, 0.2)
    q_bounds: Tuple[float, float] = (-0.2, 0.2)
    v_bounds: Tuple[float, float] = (-0.03, 0.03)


class ADMMOptConfig(BaseModel):
    iterations: int = 10
    n_simulations: int = 10
    solver_non_convex: bool = True
    time_limit: int = 10


class LongTermUncertainty(BaseModel):
    n_stages: int = 3
    number_of_scenarios: int = 100
    δ_load_var: float = 0.1
    δ_pv_var: float = 0.1
    δ_b_var: float = 0.1


class SDDPConfig(BaseModel):
    iterations: int = 10
    n_simulations: int = 100


class SDDPParams(BaseModel):
    initial_budget: float = 1e6
    discount_rate: float = 0.05
    years_per_stage: int = 1
    risk_measure_type: RiskMeasureType = RiskMeasureType.EXPECTATION
    risk_measure_param: float = 0.1
    expansion_line_cost_per_km_kw: float = 1e3
    expansion_transformer_cost_per_kw: float = 1e3
    penalty_cost_per_consumption_kw: float = 1e3
    penalty_cost_per_production_kw: float = 1e3


class ExpansionInput(BaseModel):
    grid: GridModel
    short_term_uncertainty: ShortTermUncertainty
    long_term_uncertainty: LongTermUncertainty
    admm_config: ADMMOptConfig
    sddp_config: SDDPConfig
    sddp_params: SDDPParams
    seed: int = 42


class ExpansionOutput(ExpansionResponse):
    pass


def run_expansion(input: ExpansionInput, with_ray: bool) -> ExpansionOutput:
    _, grid_data = get_grid_case(
        GridCaseModel(
            grid_case=input.grid.kace,
            s_base=input.grid.s_base,
            p_bounds=input.short_term_uncertainty.p_bounds,
            q_bounds=input.short_term_uncertainty.q_bounds,
            v_bounds=input.short_term_uncertainty.v_bounds,
            number_of_random_scenarios=input.short_term_uncertainty.number_of_scenarios,
            v_min=input.grid.v_min,
            v_max=input.grid.v_max,
            taps=input.grid.taps,
            seed=input.seed,
        )
    )
    expansion_algorithm = ExpansionAlgorithm(
        grid_data=grid_data,
        cache_dir=Path(".cache"),
        admm_groups=input.grid.groups,
        iterations=input.admm_config.iterations,
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
        with_ray=with_ray,
    )
    result = expansion_algorithm.run_pipeline()
    return ExpansionOutput(**result.model_dump())
