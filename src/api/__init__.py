from examples import *
from enum import Enum
from typing import Tuple
from pydantic import BaseModel


class GridCase(Enum):
    SIMPLE_GRID = "simple_grid"
    BOISY_GRID = "boisy_grid"
    BOISY_SIMPLIFIED = "boisy_simplified"
    ESTAVAYER_GRID = "estavayer_grid"
    ESTAVAYER_SIMPLIFIED = "estavayer_simplified"


def get_grid_case(grid_case: GridCase) -> Tuple[NodeEdgeModel, pp.pandapowerNet]:
    match grid_case:
        case GridCase.SIMPLE_GRID:
            net = pp.from_pickle("data/simple_grid.p")
            base_grid_data = pandapower_to_dig_a_plan_schema(net)
        case GridCase.BOISY_GRID:
            net = pp.from_pickle("data/boisy_grid.p")
        case GridCase.BOISY_SIMPLIFIED:
            net = pp.from_pickle("data/boisy_simplified.p")
        case GridCase.ESTAVAYER_GRID:
            net = pp.from_pickle("data/estavayer_grid.p")
        case GridCase.ESTAVAYER_SIMPLIFIED:
            net = pp.from_pickle("data/estavayer_simplified.p")
    return base_grid_data, net
