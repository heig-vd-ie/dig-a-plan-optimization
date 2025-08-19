from typing import List, Dict, Union
from pydantic import BaseModel, ConfigDict
from enum import Enum


class RiskMeasureType(str, Enum):
    EXPECTATION = "Expectation"
    ENTROPIC = "Entropic"
    WASSERSTEIN = "Wasserstein"
    WORST_CASE = "WorstCase"
    CVAR = "CVaR"


class Scenario(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    δ_load: Dict[str, float]
    δ_pv: Dict[str, float]
    δ_b: Union[float, int]


class Scenarios(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    Ω: List[List[Scenario]]
    P: List[float]


class BenderCut(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    λ_load: Dict[str, float]
    λ_pv: Dict[str, float]
    pv0: Dict[str, int]
    load0: Dict[str, int]
    λ_cap: Dict[str, int]
    cap0: Dict[str, int]
    θ: int


class BenderCuts(BaseModel):
    cuts: Dict[str, BenderCut]


class Node(BaseModel):
    id: int


class Edge(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    id: int
    target: int
    source: int


class Cut(BaseModel):
    id: int


class Grid(BaseModel):
    nodes: List[Node]
    edges: List[Edge]
    cuts: List[Cut]
    external_grid: int
    initial_cap: Dict[str, float]
    load: Dict[str, int]
    pv: Dict[str, float]
    investment_costs: Dict[str, float]
    penalty_costs_load: Dict[str, float]
    penalty_costs_pv: Dict[str, float]


class PlanningParams(BaseModel):
    n_stages: int
    initial_budget: int
    γ_cuts: float
    discount_rate: float


class AdditionalParams(BaseModel):
    iteration_limit: int
    n_simulations: int
    risk_measure_type: RiskMeasureType
    risk_measure_param: float
    seed: int


class OptimizationConfig(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    grid: Grid
    scenarios: str
    bender_cuts: str
    planning_params: PlanningParams
    additional_params: AdditionalParams


class ExpansionRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    optimization: OptimizationConfig
    scenarios: Scenarios
    bender_cuts: BenderCuts
