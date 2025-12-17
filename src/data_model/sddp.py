from typing import List, Dict, Union
from pydantic import BaseModel, ConfigDict, Field
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
    pv0: Dict[str, float]
    load0: Dict[str, float]
    λ_cap: Dict[str, float]
    cap0: Dict[str, float]
    θ: float


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
    load: Dict[str, float]
    pv: Dict[str, float]
    investment_costs: Dict[str, float]
    penalty_costs_load: Dict[str, float]
    penalty_costs_pv: Dict[str, float]
    penalty_costs_infeasibility: float


class PlanningParams(BaseModel):
    n_stages: int
    initial_budget: float
    γ_cuts: float
    discount_rate: float
    years_per_stage: int
    n_cut_scenarios: int


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
    out_of_sample_scenarios: str
    bender_cuts: str
    planning_params: PlanningParams
    additional_params: AdditionalParams


class SDDPRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    optimization: OptimizationConfig
    scenarios: Scenarios
    out_of_sample_scenarios: Scenarios
    bender_cuts: BenderCuts


class SDDPScenarioRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    n_scenarios: int
    n_stages: int
    nodes: List[Node]
    load_potential: Dict[int, float]
    pv_potential: Dict[int, float]
    min_load: float
    min_pv: float
    yearly_budget: float
    N_years_per_stage: int
    seed_number: int


class StateVar(BaseModel):
    in_: float = Field(alias="in")
    out: float


class SDDPSimulation(BaseModel):
    bellman_term: float
    node_index: int
    objective_state: Union[float, None]
    belief: Dict[str, float]
    investment_cost: float
    stage_objective: float
    cap: List[StateVar]
    obj: float
    total_unmet_load: List[StateVar]
    total_unmet_pv: List[StateVar]
    δ_cap: List[float]
    δ_load: List[float]
    δ_pv: List[float]
    δ_b: float


class SDDPResponse(BaseModel):
    objectives: List[float]
    simulations: List[List[SDDPSimulation]]
    out_of_sample_simulations: List[List[SDDPSimulation]]
    out_of_sample_objectives: List[float]
