from typing import Tuple

from pydantic import BaseModel
from pipelines.expansion.models.response import ExpansionResponse
from pipelines.expansion.models.request import RiskMeasureType
from pydantic import BaseModel, Field
from typing import Dict, List, Tuple


class GridCaseModel(BaseModel):
    pp_file: str = Field(
        default="examples/simple_grid.p", description="Path to pandapower .p file"
    )
    s_base: float = Field(default=1e6, description="Rated power in Watts")


class ShortTermUncertainty(BaseModel):
    number_of_scenarios: int = Field(
        default=10, description="Number of short term scenarios"
    )
    p_bounds: Tuple[float, float] = Field(
        default=(-0.2, 0.2), description="Active power bounds in per unit"
    )
    q_bounds: Tuple[float, float] = Field(
        default=(-0.2, 0.2), description="Reactive power bounds in per unit"
    )
    v_bounds: Tuple[float, float] = Field(
        default=(-0.03, 0.03), description="Voltage bounds in per unit"
    )


class ADMMOptConfig(BaseModel):
    iterations: int = Field(default=10, description="Number of iterations")
    n_simulations: int = Field(
        default=10, description="Number of simulations per stage"
    )
    solver_non_convex: bool = Field(default=True, description="Use non-convex solver")
    time_limit: int = Field(default=10, description="Time limit in seconds")
    groups: int | Dict[int, List[int]] = Field(
        default=10, description="Number of groups for ADMM"
    )


class LongTermUncertainty(BaseModel):
    n_stages: int = Field(default=3, description="Number of stages")
    number_of_scenarios: int = Field(
        default=100, description="Number of long-term scenarios"
    )
    δ_load_var: float = Field(default=0.1, description="Load variation in per unit")
    δ_pv_var: float = Field(default=0.1, description="PV variation in per unit")
    δ_b_var: float = Field(default=10e3, description="Budget variation in k$")


class SDDPConfig(BaseModel):
    iterations: int = Field(default=10, description="Number of iterations")
    n_simulations: int = Field(default=100, description="Number of simulations")


class ADMMParams(BaseModel):
    big_m: float = Field(default=1e3, description="Big M parameter")
    ε: float = Field(default=1e-4, description="ADMM epsilon")
    ρ: float = Field(default=2.0, description="ADMM rho")
    γ_infeasibility: float = Field(default=1.0, description="ADMM gamma infeasibility")
    γ_admm_penalty: float = Field(default=1.0, description="ADMM gamma ADMM penalty")
    γ_trafo_loss: float = Field(default=1e2, description="ADMM gamma transformer loss")
    μ: float = Field(default=10.0, description="ADMM mu")
    τ_incr: float = Field(default=2.0, description="ADMM tau increment")
    τ_decr: float = Field(default=2.0, description="ADMM tau decrement")
    voll: float = Field(default=1.0, description="ADMM value of load curtailment")
    volp: float = Field(default=1.0, description="ADMM value of production curtailment")


class SDDPParams(BaseModel):
    initial_budget: float = Field(default=50e3, description="Initial budget in k$")
    discount_rate: float = Field(default=0.05, description="Discount rate in per unit")
    years_per_stage: int = Field(default=1, description="Years per stage")
    risk_measure_type: RiskMeasureType = Field(
        default=RiskMeasureType.EXPECTATION, description="Risk measure type"
    )
    risk_measure_param: float = Field(default=0.1, description="Risk measure parameter")
    expansion_line_cost_per_km_kw: float = Field(
        default=0.2, description="Expansion line cost in k$ per km per kW"
    )
    expansion_transformer_cost_per_kw: float = Field(
        default=0.15, description="Expansion transformer cost in k$ per kW"
    )
    penalty_cost_per_consumption_kw: float = Field(
        default=0.05, description="Penalty cost in k$ per consumption per kW"
    )
    penalty_cost_per_production_kw: float = Field(
        default=0.05, description="Penalty cost in k$ per production per kW"
    )


class ExpansionInput(BaseModel):
    grid: GridCaseModel = Field(description="Grid model")
    short_term_uncertainty: ShortTermUncertainty = Field(
        description="Short term uncertainty model"
    )
    long_term_uncertainty: LongTermUncertainty = Field(
        description="Long term uncertainty model"
    )
    admm_config: ADMMOptConfig = Field(description="ADMM configuration")
    sddp_config: SDDPConfig = Field(description="SDDP configuration")
    admm_params: ADMMParams = Field(description="ADMM parameters")
    sddp_params: SDDPParams = Field(description="SDDP parameters")
    iterations: int = Field(default=10, description="Pipeline iteration numbers")
    seed: int = Field(default=42, description="Random seed")
    each_task_memory: float = Field(
        default=1e8, description="Memory allocated for each task in bytes"
    )


class InputObject(BaseModel):
    expansion: ExpansionInput
    time_now: str
    with_ray: bool


class ExpansionOutput(BaseModel):
    sddp_response: ExpansionResponse


class ADMMInput(BaseModel):
    grid: GridCaseModel = GridCaseModel()
    groups: int | dict[int, list[int]] = 10
    max_iters: int = 10
    scenarios: ShortTermUncertainty = ShortTermUncertainty()
    seed: int


class BenderInput(BaseModel):
    grid: GridCaseModel = GridCaseModel()
    max_iters: int = 100
    scenarios: ShortTermUncertainty = ShortTermUncertainty()
    seed: int = 42


class CombinedInput(BaseModel):
    grid: GridCaseModel = GridCaseModel()
    groups: int | None = None
    scenarios: ShortTermUncertainty = ShortTermUncertainty()
    seed: int = 42


class ReconfigurationOutput(BaseModel):
    switches: list[dict]
    voltages: list[dict]
    currents: list[dict]
    taps: list[dict]
