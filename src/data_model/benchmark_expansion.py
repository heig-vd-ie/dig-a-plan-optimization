from pydantic import BaseModel

from data_model import GridCaseModel, ShortTermUncertaintyProfile


class CongestionSettings(BaseModel):
    threshold: float
    max_rounds: int
    line_cost_per_km_kw: float
    trafo_cost_per_kw: float
    discount_rate: float
    years: list[int]


class BenchmarkExpansion(BaseModel):
    grid: GridCaseModel
    profiles: ShortTermUncertaintyProfile
    congestion_settings: CongestionSettings
    iterations: int
    seed: int
