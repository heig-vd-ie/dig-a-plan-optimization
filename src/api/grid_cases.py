import pandapower as pp
import polars as pl
from polars import col as c
from typing import Tuple

from data_exporter.pp_to_dap import (
    pandapower_to_dig_a_plan_schema_with_scenarios,
)
from data_model import (
    NodeEdgeModel,
    GridCaseModel,
    ShortTermUncertainty,
    ShortTermUncertaintyProfile,
)


def get_grid_case(
    grid: GridCaseModel, seed: int, stu: ShortTermUncertainty
) -> Tuple[pp.pandapowerNet, NodeEdgeModel]:
    """
    Load a pandapower grid and build the Dig-A-Plan NodeEdgeModel with scenarios.

    - Loads .p files based on GridCase
    - Calls pandapower_to_dig_a_plan_schema_with_scenarios to:
        * create NodeEdgeModel
        * generate random scenarios
    - Cleans / normalizes edge_data columns (b_pu, r_pu, x_pu, normal_open)
    """

    if isinstance(stu, ShortTermUncertaintyProfile):
        raise NotImplementedError(
            "For ShortTermUncertaintyProfile, this method is not implemented yet"
        )

    # 1) Load the pandapower network from pickle depending on the selected case
    net = pp.from_pickle(grid.pp_file)

    # 2) Build Dig-A-Plan schema + scenarios
    base_grid_data = pandapower_to_dig_a_plan_schema_with_scenarios(
        net=net,
        s_base=grid.s_base,
        v_bounds=stu.v_bounds,
        p_bounds=stu.p_bounds,
        q_bounds=stu.q_bounds,
        number_of_random_scenarios=stu.n_scenarios,
        seed=seed,
    )

    # 3) Clean / normalize edge columns for all but the simple grid
    base_grid_data.edge_data = base_grid_data.edge_data.with_columns(
        pl.when(c(col) < 1e-3).then(pl.lit(0)).otherwise(c(col)).alias(col)
        for col in ["b_pu", "r_pu", "x_pu"]
    ).with_columns(
        c("normal_open").fill_null(False),
    )

    return net, base_grid_data
