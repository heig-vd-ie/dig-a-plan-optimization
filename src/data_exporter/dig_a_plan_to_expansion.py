from typing import Dict
from pathlib import Path
from polars import col as c
from pipelines.expansion.models.request import (
    Node,
    Edge,
    Cut,
    Grid,
    BenderCuts,
    OptimizationConfig,
    PlanningParams,
    AdditionalParams,
    Scenarios,
    ExpansionRequest,
)
from data_schema import NodeEdgeModel


def dig_a_plan_to_expansion(
    grid_data: NodeEdgeModel,
    planning_params: PlanningParams,
    additional_params: AdditionalParams,
    scenarios_data: Scenarios,
    investment_costs: int | float | Dict[str, float] = 1000,
    penalty_costs_load: int | float | Dict[str, float] = 1000,
    penalty_costs_pv: int | float | Dict[str, float] = 1000,
    bender_cuts: BenderCuts | None = None,
    scenarios_cache: Path | None = None,
    bender_cuts_cache: Path | None = None,
) -> ExpansionRequest:

    nodes = [
        Node(id=node["node_id"]) for node in grid_data.node_data.iter_rows(named=True)
    ]
    edges = [
        Edge(source=edge["u_of_edge"], target=edge["v_of_edge"], id=edge["edge_id"])
        for edge in grid_data.edge_data.iter_rows(named=True)
    ]
    cuts = (
        [Cut(id=int(cut_id)) for cut_id in bender_cuts.cuts.keys()]
        if bender_cuts
        else []
    )
    external_grid = grid_data.node_data.filter(c("type") == "slack").get_column(
        "node_id"
    )[0]
    initial_cap = {
        str(edge["edge_id"]): float(edge["p_max_pu"])
        for edge in grid_data.edge_data.iter_rows(named=True)
    }
    load = {
        str(node["node_id"]): node["cons_installed"]
        for node in grid_data.node_data.iter_rows(named=True)
    }
    prod = {
        str(node["node_id"]): node["prod_installed"]
        for node in grid_data.node_data.iter_rows(named=True)
    }

    if isinstance(investment_costs, (int, float)):
        investment_costs = {
            str(edge["edge_id"]): float(investment_costs)
            for edge in grid_data.edge_data.iter_rows(named=True)
        }
    if isinstance(penalty_costs_load, (int, float)):
        penalty_costs_load = {
            str(edge["edge_id"]): float(penalty_costs_load)
            for edge in grid_data.edge_data.iter_rows(named=True)
        }
    if isinstance(penalty_costs_pv, (int, float)):
        penalty_costs_pv = {
            str(edge["edge_id"]): float(penalty_costs_pv)
            for edge in grid_data.edge_data.iter_rows(named=True)
        }

    grid = Grid(
        nodes=nodes,
        edges=edges,
        cuts=cuts,
        external_grid=external_grid,
        initial_cap=initial_cap,
        load=load,
        pv=prod,
        investment_costs=investment_costs,
        penalty_costs_load=penalty_costs_load,
        penalty_costs_pv=penalty_costs_pv,
    )

    return ExpansionRequest(
        optimization=OptimizationConfig(
            grid=grid,
            scenarios=(
                str(scenarios_cache) if scenarios_cache else ".cache/scenarios.json"
            ),
            bender_cuts=(
                str(bender_cuts_cache)
                if bender_cuts_cache
                else ".cache/bender_cuts.json"
            ),
            planning_params=planning_params,
            additional_params=additional_params,
        ),
        scenarios=scenarios_data,
        bender_cuts=bender_cuts if bender_cuts else BenderCuts(cuts={}),
    )
