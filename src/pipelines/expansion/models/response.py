from typing import List, Union, Dict
from pydantic import BaseModel, Field


class StateVar(BaseModel):
    in_: float = Field(alias="in")
    out: float


class NoiseTerm(BaseModel):
    δ_load: StateVar
    δ_pv: StateVar
    δ_b: float


class Simulation(BaseModel):
    bellman_term: float
    node_index: int
    objective_state: Union[float, None]
    belief: Dict[str, float]
    investment_cost: float
    stage_objective: float
    cap: List[StateVar]
    obj: float
    total_unmet_load: List[StateVar]


class ExpansionResponse(BaseModel):
    objectives: List[float]
    simulations: List[List[Simulation]]
