from pathlib import Path
from enum import Enum
from pydantic import BaseModel
from pydantic import BaseModel, Field
from data_model.kace import GridCaseModel, LoadProfiles
from data_model.reconfiguration_configs import ADMMConfig, BenderConfig, CombinedConfig


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


class ADMMInput(BaseModel):
    grid: GridCaseModel = GridCaseModel()
    konfig: ADMMConfig = ADMMConfig()
    scenarios: ShortTermScenarios = ShortTermScenarios()
    load_profiles: LoadProfiles = LoadProfiles()
    save_path: Path | None = None
    seed: int = 42


class BenderInput(BaseModel):
    grid: GridCaseModel = GridCaseModel()
    konfig: BenderConfig = BenderConfig()
    scenarios: ShortTermScenarios = ShortTermScenarios()
    load_profiles: LoadProfiles = LoadProfiles()
    seed: int = 42


class CombinedInput(BaseModel):
    grid: GridCaseModel = GridCaseModel()
    konfig: CombinedConfig = CombinedConfig()
    scenarios: ShortTermScenarios = ShortTermScenarios()
    load_profiles: LoadProfiles = LoadProfiles()
    seed: int = 42


class ReconfigurationOutput(BaseModel):
    switches: list[dict]
    voltages: list[dict]
    currents: list[dict]
    taps: list[dict]
