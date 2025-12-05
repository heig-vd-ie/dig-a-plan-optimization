from enum import Enum
from typing import Tuple

from pydantic import BaseModel


class GridCase(Enum):
    SIMPLE_GRID = "simple_grid"
    BOISY_GRID = "boisy_grid"
    BOISY_SIMPLIFIED = "boisy_simplified"
    ESTAVAYER_GRID = "estavayer_grid"
    ESTAVAYER_SIMPLIFIED = "estavayer_simplified"


class GridCaseModel(BaseModel):
    grid_case: GridCase
    s_base: float = 1e6
    taps: list[int] = list(range(95, 105, 1))
    p_bounds: Tuple[float, float] = (-0.2, 0.2)
    q_bounds: Tuple[float, float] = (-0.2, 0.2)
    v_bounds: Tuple[float, float] = (-0.03, 0.03)
    number_of_random_scenarios: int = 10
    v_min: float = 0.9
    v_max: float = 1.1
    seed: int = 42


class ReconfigurationOutput(BaseModel):
    switches: list[dict]
    voltages: list[dict]
    currents: list[dict]
    taps: list[dict]
