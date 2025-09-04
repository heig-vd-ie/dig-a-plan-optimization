from typing import List, Union, Dict
from pydantic import BaseModel, Field


class StateVar(BaseModel):
    in_: float = Field(alias="in")
    out: float


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
    total_unmet_pv: List[StateVar]
    δ_cap: List[float]
    δ_load: List[float]
    δ_pv: List[float]
    δ_b: float


class ExpansionResponse(BaseModel):
    objectives: List[float]
    simulations: List[List[Simulation]]
    out_of_sample_simulations: List[List[Simulation]]
    out_of_sample_objectives: List[float]
