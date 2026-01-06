from pydantic import BaseModel
from pydantic import BaseModel, Field
from typing import Dict, List, Tuple
from data_model.kace import GridCaseModel


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
    solver_non_convex: bool = Field(default=True, description="Use non-convex solver")
    time_limit: int = Field(default=10, description="Time limit in seconds")
    groups: int | Dict[int, List[int]] = Field(
        default=10, description="Number of groups for ADMM"
    )


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


class ReconfigurationOutput(BaseModel):
    switches: list[dict]
    voltages: list[dict]
    currents: list[dict]
    taps: list[dict]


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
