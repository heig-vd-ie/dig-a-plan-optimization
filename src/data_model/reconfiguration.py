from pathlib import Path
from enum import Enum
from pydantic import BaseModel
from pydantic import BaseModel, Field
from data_model.kace import GridCaseModel, LoadProfiles


class DiscreteScenario(Enum):
    BASIC = "Basic"
    SUSTAINABLE = "Sustainable"
    FULL = "Full"


class ShortTermScenarios(BaseModel):
    target_year: int = Field(default=2030, description="Target year for scenarios")
    quarter: int = Field(ge=1, le=4, default=1, description="Quarter of the year (1-4)")
    scenario_name: DiscreteScenario = Field(
        default=DiscreteScenario.BASIC, description="Type of discrete scenario"
    )
    n_scenarios: int = Field(default=10, description="Number of scenarios")


class ADMMParams(BaseModel):
    groups: int | dict[int, list[int]] = 10
    max_iters: int = 10
    solver_non_convex: bool = Field(default=True, description="Use non-convex solver")
    time_limit: int = Field(default=10, description="Time limit in seconds")
    ε: float = Field(default=1e-4, description="ADMM epsilon")
    ρ: float = Field(default=2.0, description="ADMM rho")
    γ_infeasibility: float = Field(default=1.0, description="ADMM gamma infeasibility")
    γ_admm_penalty: float = Field(default=1.0, description="ADMM gamma ADMM penalty")
    γ_trafo_loss: float = Field(default=1e2, description="ADMM gamma transformer loss")
    μ: float = Field(default=10.0, description="ADMM mu")
    τ_incr: float = Field(default=2.0, description="ADMM tau increment")
    τ_decr: float = Field(default=2.0, description="ADMM tau decrement")
    big_m: float = Field(default=1e3, description="Big M parameter")
    voll: float = Field(default=1.0, description="ADMM value of load curtailment")
    volp: float = Field(default=1.0, description="ADMM value of production curtailment")


class ADMMInput(BaseModel):
    grid: GridCaseModel = GridCaseModel()
    params: ADMMParams = ADMMParams()
    scenarios: ShortTermScenarios = ShortTermScenarios()
    load_profiles: LoadProfiles = LoadProfiles()
    save_path: Path | None = None
    seed: int


class BenderInput(BaseModel):
    grid: GridCaseModel = GridCaseModel()
    max_iters: int = 100
    scenarios: ShortTermScenarios = ShortTermScenarios()
    load_profiles: LoadProfiles = LoadProfiles()
    seed: int = 42


class CombinedInput(BaseModel):
    grid: GridCaseModel = GridCaseModel()
    groups: int | None = None
    scenarios: ShortTermScenarios = ShortTermScenarios()
    load_profiles: LoadProfiles = LoadProfiles()
    seed: int = 42


class ReconfigurationOutput(BaseModel):
    switches: list[dict]
    voltages: list[dict]
    currents: list[dict]
    taps: list[dict]
