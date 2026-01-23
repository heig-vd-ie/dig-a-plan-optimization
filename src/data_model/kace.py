from enum import Enum
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Tuple, Union


class DiscreteScenario(Enum):
    BASIC = "Basic"
    SUSTAINABLE = "Sustainable"
    FULL = "Full"


class GridCaseModel(BaseModel):
    name: str = Field(default="default", description="Grid case name")
    pp_file: str = Field(
        default="examples/ieee_33/simple_grid.p",
        description="Path to pandapower .p file",
    )
    s_base: float = Field(default=1e6, description="Rated power in Watts")
    cosφ: float = Field(default=0.95, description="Power factor")
    egid_id_mapping_file: str = Field(
        default="examples/ieee_33/consumer_egid_idx_mapping.csv",
        description="Path to EGID to ID mapping CSV file",
    )
    minimum_impedance: float = Field(
        default=0.0, description="Minimum impedance to avoid numerical issues"
    )


class ShortTermUncertintyBase(BaseModel):
    v_bounds: Tuple[float, float] = Field(
        default=(-0.03, 0.03), description="Voltage bounds in per unit"
    )
    n_scenarios: int = Field(default=10, description="Number of short term scenarios")


class ShortTermUncertaintyProfile(ShortTermUncertintyBase):
    load_profiles: list[str] = Field(
        default=["examples/ieee_33/load_profiles"],
        description="List of paths to load profile directories",
    )
    pv_profile: str = Field(
        default="examples/ieee_33/pv_profiles",
        description="Path to PV profile directory",
    )
    target_year: int = Field(default=2025, description="Target year for scenarios")
    quarter: int = Field(ge=1, le=4, default=1, description="Quarter of the year (1-4)")
    scenario_name: DiscreteScenario = Field(
        default=DiscreteScenario.BASIC, description="Type of discrete scenario"
    )


class ShortTermUncertaintyRandom(ShortTermUncertintyBase):
    p_bounds: Tuple[float, float] = Field(
        default=(-0.2, 0.2), description="Active power bounds in per unit"
    )
    q_bounds: Tuple[float, float] = Field(
        default=(-0.2, 0.2), description="Reactive power bounds in per unit"
    )
