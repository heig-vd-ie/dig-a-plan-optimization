from typing import Tuple

import pandapower as pp
import polars as pl
from polars import col as c
# from experiments import *
from experiments import (
    pandapower_to_dig_a_plan_schema_with_scenarios,
    NodeEdgeModel,
)

from .models import GridCase, GridCaseModel


def get_grid_case(input: GridCaseModel) -> Tuple[pp.pandapowerNet, NodeEdgeModel]:
    """
    Load a pandapower grid and build the Dig-A-Plan NodeEdgeModel with scenarios.

    - Loads .p files based on GridCase
    - Calls pandapower_to_dig_a_plan_schema_with_scenarios to:
        * create NodeEdgeModel
        * generate random scenarios
    - Cleans / normalizes edge_data columns (b_pu, r_pu, x_pu, normal_open)
    """

    # 1) Load the pandapower network from pickle depending on the selected case
    match input.grid_case:
        case GridCase.SIMPLE_GRID:
            net = pp.from_pickle("examples/simple_grid.p")
        case GridCase.BOISY_GRID:
            net = pp.from_pickle(".cache/input/boisy/boisy_grid.p")
        case GridCase.BOISY_SIMPLIFIED:
            net = pp.from_pickle(".cache/input/boisy/boisy_grid_simplified.p")
        case GridCase.ESTAVAYER_GRID:
            net = pp.from_pickle(".cache/input/estavayer/estavayer_grid.p")
        case GridCase.ESTAVAYER_SIMPLIFIED:
            net = pp.from_pickle(".cache/input/estavayer/estavayer_grid_simplified.p")

    # 2) Build Dig-A-Plan schema + scenarios
    base_grid_data = pandapower_to_dig_a_plan_schema_with_scenarios(
        net=net,
        s_base=input.s_base,
        taps=input.taps,
        v_bounds=input.v_bounds,
        p_bounds=input.p_bounds,
        q_bounds=input.q_bounds,
        number_of_random_scenarios=input.number_of_random_scenarios,
        v_min=input.v_min,
        v_max=input.v_max,
        seed=input.seed,
    )

    # 3) Clean / normalize edge columns for all but the simple grid
    if input.grid_case != GridCase.SIMPLE_GRID:
        base_grid_data.edge_data = (
            base_grid_data.edge_data.with_columns(
                pl.when(c(col) < 1e-3)
                .then(pl.lit(0))
                .otherwise(c(col))
                .alias(col)
                for col in ["b_pu", "r_pu", "x_pu"]
            ).with_columns(
                c("normal_open").fill_null(False),
            )
        )

    return net, base_grid_data
