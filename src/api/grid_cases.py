import pandapower as pp
import polars as pl
from polars import col as c
from typing import Tuple

from data_exporter.pp_to_dap import (
    pp_to_dap,
)
from data_exporter.uncert_to_scens_prof import (
    generate_profile_based_load_scenarios,
)
from data_exporter.uncert_to_scens_rand import generate_random_load_scenarios
from data_model import (
    NodeEdgeModel,
    GridCaseModel,
    ShortTermUncertaintyRandom,
    ShortTermUncertaintyProfile,
)


def fill_missing_bus_geo(net: pp.pandapowerNet) -> pp.pandapowerNet:

    missing = net.bus["geo"].isna()
    idxs = net.bus.index[missing]

    for k, idx in enumerate(idxs):
        net.bus.at[idx, "geo"] = f'{{"type":"Point","coordinates":[{float(k)},{0.0}]}}'

    print("Filled missing geo:", len(idxs))
    return net


def get_grid_case(
    grid: GridCaseModel,
    seed: int,
    stu: ShortTermUncertaintyRandom,
    profiles: ShortTermUncertaintyProfile | None = None,
) -> Tuple[pp.pandapowerNet, NodeEdgeModel]:
    """
    Load a pandapower grid and build the Dig-A-Plan NodeEdgeModel with scenarios.

    - Loads .p files based on GridCase
    - Calls pp_to_dap to:
        * create NodeEdgeModel
        * generate random scenarios
    - Cleans / normalizes edge_data columns (b_pu, r_pu, x_pu, normal_open)
    """

    # 1) Load the pandapower network from pickle depending on the selected case
    net = pp.from_pickle(grid.pp_file)
    net = fill_missing_bus_geo(net)

    node_edge_model, load_data, v_slack_node_sqr_pu = pp_to_dap(net, s_base=grid.s_base)

    # 2) Build Dig-A-Plan schema + scenarios
    if profiles is not None:
        rand_scenarios = generate_profile_based_load_scenarios(
            grid=grid,
            profiles=profiles,
            net=net,
            seed=seed,
        )
    else:
        rand_scenarios = generate_random_load_scenarios(
            node_edge_model=node_edge_model,
            load_data=load_data,
            stu=stu,
            v_slack_node_sqr_pu=v_slack_node_sqr_pu,
            seed=seed,
        )

    node_edge_model.load_data = rand_scenarios

    node_edge_model.edge_data = node_edge_model.edge_data.with_columns(
        pl.when(c(col) < 1e-3).then(pl.lit(0)).otherwise(c(col)).alias(col)
        for col in ["b_pu", "r_pu", "x_pu"]
    ).with_columns(
        c("normal_open").fill_null(False),
    )

    return net, node_edge_model
