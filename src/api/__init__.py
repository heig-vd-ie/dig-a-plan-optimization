from examples import *
from enum import Enum
from typing import Tuple
from pydantic import BaseModel
import ray
import socket


@ray.remote
def where_am_i():
    return socket.gethostname(), ray.util.get_node_ip_address()


class GridCase(Enum):
    SIMPLE_GRID = "simple_grid"
    BOISY_GRID = "boisy_grid"
    BOISY_SIMPLIFIED = "boisy_simplified"
    ESTAVAYER_GRID = "estavayer_grid"
    ESTAVAYER_SIMPLIFIED = "estavayer_simplified"


class GridCaseModel(BaseModel):
    grid_case: GridCase
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


def get_grid_case(input: GridCaseModel) -> Tuple[pp.pandapowerNet, NodeEdgeModel]:
    match input.grid_case:
        case GridCase.SIMPLE_GRID:
            net = pp.from_pickle("data/simple_grid.p")
        case GridCase.BOISY_GRID:
            net = pp.from_pickle(".cache/input/boisy/boisy_grid.p")
        case GridCase.BOISY_SIMPLIFIED:
            net = pp.from_pickle(".cache/input/boisy/boisy_grid_simplified.p")
        case GridCase.ESTAVAYER_GRID:
            net = pp.from_pickle(".cache/input/estavayer/estavayer_grid.p")
        case GridCase.ESTAVAYER_SIMPLIFIED:
            net = pp.from_pickle(".cache/input/estavayer/estavayer_simplified.p")
    base_grid_data = pandapower_to_dig_a_plan_schema(
        net=net,
        taps=input.taps,
        v_bounds=input.v_bounds,
        p_bounds=input.p_bounds,
        q_bounds=input.q_bounds,
        number_of_random_scenarios=input.number_of_random_scenarios,
        v_min=input.v_min,
        v_max=input.v_max,
        seed=input.seed,
    )
    if input.grid_case in {
        GridCase.BOISY_SIMPLIFIED,
        GridCase.ESTAVAYER_SIMPLIFIED,
    }:
        base_grid_data.edge_data = base_grid_data.edge_data.with_columns(
            pl.when(c(col) < 1e-3).then(pl.lit(0)).otherwise(c(col)).alias(col)
            for col in ["b_pu", "r_pu", "x_pu"]
        ).with_columns(
            c("normal_open").fill_null(False),
        )

    return net, base_grid_data
