import pandapower as pp
from data_exporter.pp_to_dap import pp_to_dap_schema
from data_exporter.scenario_reduction import ScenarioPipeline
from data_model.kace import GridCaseModel
from data_model import NodeEdgeModel
from data_model.expansion import LongTermScenarios
from data_model.kace import LoadProfiles
from data_model.reconfiguration import ShortTermScenarios


def kace4reconfiguration(
    grid: GridCaseModel,
    load_profiles: LoadProfiles,
    st_scenarios: ShortTermScenarios,
    seed: int,
) -> NodeEdgeModel:
    net = pp.from_pickle(grid.pp_file)
    base_grid_data = pp_to_dap_schema(net=net, s_base=grid.s_base)
    rand_scenarios = (
        ScenarioPipeline(load_profiles)
        .process(st_scenarios=st_scenarios)
        .map2scens(
            id_node_mapping=net.load,
            cosφ=grid.cosφ,
            s_base=grid.s_base,
            seed=seed,
        )
    )
    return NodeEdgeModel(
        node_data=base_grid_data.node_data,
        edge_data=base_grid_data.edge_data,
        load_data=rand_scenarios,
    )


def kace4expansion(
    grid: GridCaseModel,
    load_profiles: LoadProfiles,
    long_term_scenarios: LongTermScenarios,
    seed: int,
) -> NodeEdgeModel:
    NotImplementedError("")
    return NodeEdgeModel()
