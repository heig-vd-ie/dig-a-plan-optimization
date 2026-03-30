from pydantic import BaseModel
from typing import Dict
from data_model import GridCaseModel, ShortTermUncertaintyProfile


class CongestionSettings(BaseModel):
    threshold: float
    voltage_limits: float
    max_rounds: int
    line_cost_per_km_kw: float
    trafo_cost_per_kw: float
    discount_rate: float
    years: list[int]
    reserve_percent: float


class BenchmarkExpansion(BaseModel):
    grid: GridCaseModel
    profiles: ShortTermUncertaintyProfile
    congestion_settings: CongestionSettings
    iterations: int
    seed: int


class PowerFlowResponse(BaseModel):
    congested_lines: list[Dict]
    congested_trafos: list[Dict]
    congested_buses: list[Dict]
