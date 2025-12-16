from enum import Enum
from pathlib import Path
import polars as pl
from pydantic import BaseModel, ConfigDict, Field


class DiscreteScenario(Enum):
    BASIC = "Basic"
    SUSTAINABLE = "Sustainable"
    FULL = "Full"


class KnownScenariosOptions(BaseModel):
    load_profiles: list[Path]
    pv_profile: Path
    target_year: int
    quarter: int = Field(ge=1, le=4)
    scenario_name: DiscreteScenario
    n_scenarios: int
