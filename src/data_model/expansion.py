from pydantic import BaseModel
from data_model.sddp import ExpansionResponse, RiskMeasureType
from pydantic import BaseModel, Field
from data_model.kace import GridCaseModel
from data_model.reconfiguration import (
    ShortTermUncertaintyRandom,
    ShortTermUncertaintyProfile,
    ADMMConfig,
)


class SDDPConfig(BaseModel):
    iterations: int = Field(default=10, description="Number of iterations")
    n_stages: int = Field(default=3, description="Number of stages")
    n_scenarios: int = Field(default=100, description="Number of long-term scenarios")
    n_simulations: int = Field(default=100, description="Number of simulations")
    n_optimizations: int = Field(
        default=10, description="Number of ADMM optimization per stage"
    )
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
    δ_b_var: float = Field(
        default=5000.0, description="Yearly budget variation for long-term scenarios"
    )


class ExpansionInput(BaseModel):
    grid: GridCaseModel = Field(description="Grid model")
    profiles: ShortTermUncertaintyProfile | None = Field(
        description="Profile-based uncertainty model", default=None
    )
    short_term_uncertainty: ShortTermUncertaintyRandom = Field(
        description="Short term uncertainty model", default=ShortTermUncertaintyRandom()
    )
    admm_config: ADMMConfig = Field(description="ADMM configuration")
    sddp_config: SDDPConfig = Field(description="SDDP configuration")
    iterations: int = Field(default=10, description="Pipeline iteration numbers")
    seed: int = Field(default=42, description="Random seed")


class ExpansionOutput(BaseModel):
    sddp_response: ExpansionResponse
