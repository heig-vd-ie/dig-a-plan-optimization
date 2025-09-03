import networkx as nx
import copy
from numpy import s_
import polars as pl
from typing import Dict
from pathlib import Path
from networkx import connected_components
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
from pipelines.helpers.json_rw import save_obj_to_json


def remove_switches_from_grid_data(grid_data: NodeEdgeModel) -> NodeEdgeModel:
    """Remove switches from the grid data."""
    graph = nx.Graph()
    for edge in grid_data.edge_data.filter(c("type") == "switch").iter_rows(named=True):
        graph.add_edge(edge["u_of_edge"], edge["v_of_edge"])
    connected_subgraphs = list(connected_components(graph))

    nodes_mapping = [
        {
            "nodes": list(s),
            "keep": (
                list(s)[0]
                if not any(
                    [
                        grid_data.node_data.filter(c("node_id") == n)["type"][0]
                        == "slack"
                        for n in s
                    ]
                )
                else grid_data.node_data.filter(c("type") == "slack")["node_id"][0]
            ),
        }
        for s in connected_subgraphs
    ]
    nodes_to_remove = [
        node
        for mapping in nodes_mapping
        for node in mapping["nodes"]
        if mapping["keep"] != node
    ]

    grid_data_rm = copy.deepcopy(grid_data)
    grid_data_rm.node_data = grid_data_rm.node_data.filter(
        ~c("node_id").is_in(nodes_to_remove)
    )
    grid_data_rm.edge_data = grid_data_rm.edge_data.filter(c("type") != "switch")
    for s in nodes_mapping:
        grid_data_rm.edge_data = grid_data_rm.edge_data.with_columns(
            c("u_of_edge")
            .map_elements(
                lambda x: s["keep"] if x in s["nodes"] else x, return_dtype=pl.Int32
            )
            .alias("u_of_edge"),
            c("v_of_edge")
            .map_elements(
                lambda x: s["keep"] if x in s["nodes"] else x, return_dtype=pl.Int32
            )
            .alias("v_of_edge"),
        )
    return grid_data_rm


def dig_a_plan_to_expansion(
    grid_data: NodeEdgeModel,
    planning_params: PlanningParams,
    additional_params: AdditionalParams,
    scenarios_data: Scenarios,
    out_of_sample_scenarios: Scenarios,
    s_base: float = 1e6,
    expansion_transformer_cost_per_kw: int | float = 1000,
    expansion_line_cost_per_km_kw: int | float = 1000,
    penalty_cost_per_consumption_kw: int | float = 1000,
    penalty_cost_per_production_kw: int | float = 1000,
    bender_cuts: BenderCuts | None = None,
    scenarios_cache: Path | None = None,
    out_of_sample_scenarios_cache: Path | None = None,
    bender_cuts_cache: Path | None = None,
    optimization_config_cache: Path | None = None,
) -> ExpansionRequest:
    """Convert Dig-A-Plan data model to expansion request."""
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

    investment_costs = {
        str(edge["edge_id"]): (
            float(expansion_line_cost_per_km_kw) * edge["length_km"] * s_base / 1e3
            if edge["type"] == "branch"
            else (
                float(expansion_transformer_cost_per_kw)
                * edge["p_max_pu"]
                * s_base
                / 1e3
                if edge["type"] == "transformer"
                else 0.0
            )
        )
        for edge in grid_data.edge_data.iter_rows(named=True)
    }
    penalty_costs_load = {
        str(node["node_id"]): float(penalty_cost_per_consumption_kw)
        * node["cons_installed"]
        * s_base
        / 1e3
        for node in grid_data.node_data.iter_rows(named=True)
    }
    penalty_costs_pv = {
        str(node["node_id"]): float(penalty_cost_per_production_kw)
        * node["prod_installed"]
        * s_base
        / 1e3
        for node in grid_data.node_data.iter_rows(named=True)
    }
    penalty_costs_infeasibility = s_base / 1e3

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
        penalty_costs_infeasibility=penalty_costs_infeasibility,
    )
    optimization_config = OptimizationConfig(
        grid=grid,
        scenarios=(
            str(scenarios_cache) if scenarios_cache else ".cache/scenarios.json"
        ),
        out_of_sample_scenarios=(
            str(out_of_sample_scenarios_cache)
            if out_of_sample_scenarios_cache
            else ".cache/out_of_sample_scenarios.json"
        ),
        bender_cuts=(
            str(bender_cuts_cache) if bender_cuts_cache else ".cache/bender_cuts.json"
        ),
        planning_params=planning_params,
        additional_params=additional_params,
    )

    if optimization_config_cache:
        save_obj_to_json(optimization_config, optimization_config_cache)

    return ExpansionRequest(
        optimization=optimization_config,
        scenarios=scenarios_data,
        out_of_sample_scenarios=out_of_sample_scenarios,
        bender_cuts=bender_cuts if bender_cuts is not None else BenderCuts(cuts={}),
    )
