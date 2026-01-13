import pandapower as pp
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
    ShortTermUncertainty,
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
    stu: ShortTermUncertainty,
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

    node_edge_model, v_slack_node_sqr_pu = pp_to_dap(net, s_base=grid.s_base)

    # 2) Build Dig-A-Plan schema + scenarios
    if isinstance(stu, ShortTermUncertaintyProfile):
        rand_scenarios = generate_profile_based_load_scenarios(
            grid=grid,
            stu=stu,
            net=net,
            seed=seed,
        )
    else:
        rand_scenarios = generate_random_load_scenarios(
            node_edge_model=node_edge_model,
            stu=stu,
            v_slack_node_sqr_pu=v_slack_node_sqr_pu,
            seed=seed,
        )

    base_grid_data = NodeEdgeModel(
        node_data=node_edge_model.node_data,
        edge_data=node_edge_model.edge_data,
        load_data=rand_scenarios,
    )

    return net, base_grid_data
