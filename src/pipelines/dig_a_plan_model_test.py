import polars as pl
from polars import col as c
import networkx as nx
import os
from math import log10
import logging

import tqdm
import numpy as np
import pyomo.environ as pyo

from pyomo.environ import Suffix
import patito as pt
from typing import TypedDict, Unpack

from general_function import pl_to_dict, generate_log, pl_to_dict_with_tuple
from polars_function import list_to_list_of_tuple, cast_boolean, modify_string_col
from networkx_function import (
    generate_nx_edge,
    generate_bfs_tree_with_edge_data,
    get_all_edge_data,
)

from data_schema.node_data import NodeData
from data_schema.edge_data import EdgeData

from optimization_model.master_model.sets import master_model_sets
from optimization_model.master_model.parameters import test_master_model_parameters
from optimization_model.master_model.variables import test_master_model_variables
from optimization_model.master_model.constraints import master_model_constraints

from optimization_model.slave_model.sets import slave_model_sets
from optimization_model.slave_model.parameters import slave_model_parameters
from optimization_model.slave_model.variables import slave_model_variables
from optimization_model.slave_model.constraints import slave_model_constraints

from itertools import combinations
from copy import deepcopy

from pyomo_utility import extract_optimization_results

log = generate_log(name=__name__)


class DataSchemaPolarsModel(TypedDict, total=True):
    node_data: pl.DataFrame
    edge_data: pl.DataFrame


def generate_master_model() -> pyo.AbstractModel:
    master_model: pyo.AbstractModel = pyo.AbstractModel()  # type: ignore
    master_model = master_model_sets(master_model)
    master_model = test_master_model_parameters(master_model)
    master_model = test_master_model_variables(master_model)
    master_model = master_model_constraints(master_model)
    return master_model


def generate_slave_model() -> pyo.AbstractModel:
    slave_model: pyo.AbstractModel = pyo.AbstractModel()  # type: ignore
    slave_model = slave_model_sets(slave_model)
    slave_model = slave_model_parameters(slave_model)
    slave_model = slave_model_variables(slave_model)
    slave_model = slave_model_constraints(slave_model)
    return slave_model


class DigAPlanTest:
    def __init__(
        self,
        verbose: bool = False,
        big_m: float = 1e4,
        penalty_cost: float = 1e3,
        current_factor: float = 1e2,
        voltage_factor: float = 1e1,
        power_factor: float = 1e1,
        slack_threshold: float = 1e-5,
        convergence_threshold=1e-4,
    ) -> None:

        self.verbose: int = verbose
        self.big_m: float = big_m
        self.convergence_threshold: float = convergence_threshold
        self.current_factor: float = current_factor
        self.voltage_factor: float = voltage_factor
        self.power_factor: float = power_factor
        self.slack_threshold: float = slack_threshold
        self.d: pl.DataFrame = pl.DataFrame()
        self.infeasible_slave: bool
        self.slave_obj: float
        self.master_obj: float = -1e8
        self.penalty_cost: float = penalty_cost

        self.__node_data: pt.DataFrame[NodeData] = NodeData.DataFrame(
            schema=NodeData.columns
        ).cast()
        self.__edge_data: pt.DataFrame[EdgeData] = EdgeData.DataFrame(
            schema=EdgeData.columns
        ).cast()
        self.__master_model: pyo.AbstractModel = generate_master_model()

        self.__slave_model: pyo.AbstractModel = generate_slave_model()
        self.__master_model_instance: pyo.ConcreteModel
        self.__slave_model_instance: pyo.ConcreteModel

        self.__slack_node: int
        self.master_solver = pyo.SolverFactory("gurobi")
        self.master_solver.options["IntegralityFocus"] = (
            1  # To insure master binary variable remains binary
        )
        self.slave_solver = pyo.SolverFactory("gurobi")
        self.slave_solver.options["NonConvex"] = 2  # To allow non-convex optimization
        self.slave_solver.options["QCPDual"] = (
            1  # To allow dual variables extraction on quadratic constraints
        )
        self.slack_i_sq: pl.DataFrame
        self.slack_v_pos: pl.DataFrame
        self.slack_v_neg: pl.DataFram

        self.switch_combination: pl.DataFrame

    @property
    def node_data(self) -> pt.DataFrame[NodeData]:
        return self.__node_data

    @property
    def edge_data(self) -> pt.DataFrame[EdgeData]:
        return self.__edge_data

    @property
    def master_model(self) -> pyo.AbstractModel:
        return self.__master_model

    @property
    def slave_model(self) -> pyo.AbstractModel:
        return self.__slave_model

    @property
    def slave_model_instance(self) -> pyo.ConcreteModel:
        return self.__slave_model_instance

    @property
    def master_model_instance(self) -> pyo.ConcreteModel:
        return self.__master_model_instance

    @property
    def slack_node(self) -> int:
        return self.__slack_node

    def __node_data_setter(self, node_data: pl.DataFrame):
        old_table: pl.DataFrame = self.__node_data.clear()
        col_list: list[str] = list(
            set(node_data.columns).intersection(set(old_table.columns))
        )
        new_table_pl: pl.DataFrame = pl.concat(
            [old_table, node_data.select(col_list)], how="diagonal_relaxed"
        )
        new_table_pt: pt.DataFrame[NodeData] = (
            pt.DataFrame(new_table_pl)
            .set_model(NodeData)
            .fill_null(strategy="defaults")
            .cast(strict=True)
        )
        new_table_pt.validate()
        self.__node_data = new_table_pt

    def __edge_data_setter(self, edge_data: pl.DataFrame):
        old_table: pl.DataFrame = self.__edge_data.clear()
        col_list: list[str] = list(
            set(edge_data.columns).intersection(set(old_table.columns))
        )
        new_table_pl: pl.DataFrame = pl.concat(
            [old_table, edge_data.select(col_list)], how="diagonal_relaxed"
        )
        new_table_pt: pt.DataFrame[EdgeData] = (
            pt.DataFrame(new_table_pl)
            .set_model(EdgeData)
            .fill_null(strategy="defaults")
            .cast(strict=True)
        )
        new_table_pt.validate()
        self.__edge_data = new_table_pt

    def __instantiate_model(self):

        grid_data = {
            None: {
                "N": {None: self.node_data["node_id"].to_list()},
                "L": {None: self.edge_data["edge_id"].to_list()},
                "C": dict(
                    zip(
                        self.edge_data["edge_id"].to_list(),
                        map(
                            lambda x: [tuple(x)],
                            self.edge_data.select(
                                pl.concat_list("u_of_edge", "v_of_edge").alias("nodes")
                            )["nodes"].to_list(),
                        ),
                    )
                ),
                "S": {
                    None: self.edge_data.filter(c("type") == "switch")[
                        "edge_id"
                    ].to_list()
                },
                "r": pl_to_dict(self.edge_data["edge_id", "r_pu"]),
                "x": pl_to_dict(self.edge_data["edge_id", "x_pu"]),
                "b": pl_to_dict(self.edge_data["edge_id", "b_pu"]),
                "n_transfo": pl_to_dict_with_tuple(
                    self.edge_data.select(
                        pl.concat_list("edge_id", "u_of_edge", "v_of_edge"), "n_transfo"
                    )
                ),
                "p_node": pl_to_dict(self.node_data["node_id", "p_node_pu"]),
                "q_node": pl_to_dict(self.node_data["node_id", "q_node_pu"]),
                "i_max": pl_to_dict(self.edge_data["edge_id", "i_max_pu"]),
                "v_min": pl_to_dict(self.node_data["node_id", "v_min_pu"]),
                "v_max": pl_to_dict(self.node_data["node_id", "v_max_pu"]),
                "slack_node": {None: self.slack_node},
                "slack_node_v_sq": {
                    None: self.node_data.filter(c("type") == "slack")["v_node_sqr_pu"][
                        0
                    ]
                },
                "big_m": {None: self.big_m},
                "penalty_cost": {None: self.penalty_cost},
                "current_factor": {None: self.current_factor},
                "voltage_factor": {None: self.voltage_factor},
                "power_factor": {None: self.power_factor},
            }
        }

        self.__master_model_instance = self.master_model.create_instance(grid_data)  # type: ignore
        self.__slave_model_instance = self.slave_model.create_instance(grid_data)  # type: ignore

    def add_grid_data(self, **grid_data: Unpack[DataSchemaPolarsModel]) -> None:

        for table_name, pl_table in grid_data.items():
            if table_name == "node_data":
                self.__node_data_setter(node_data=pl_table)  # type: ignore
            elif table_name == "edge_data":
                self.__edge_data_setter(edge_data=pl_table)  # type: ignore
            else:
                raise ValueError(f"{table_name} is not a valid name")

        if self.node_data.filter(c("type") == "slack").height != 1:
            raise ValueError("There must be only one slack node")

        self.__slack_node: int = self.node_data.filter(c("type") == "slack")["node_id"][
            0
        ]
        self.__instantiate_model()

    def test_all_switch_combinations(self) -> None:
        switch_list = self.edge_data.filter(c("type") == "switch")["edge_id"].to_list()

        master_feasible = []
        slave_results = []

        switch_combination = pl.DataFrame(
            list(map(list, combinations(switch_list, 5))),
            schema=[f"switch_{n}" for n in range(5)],
            orient="row",
        )
        for row in tqdm.tqdm(
            switch_combination.rows(), desc="Checking switch combinations"
        ):
            open_switches = dict(zip(row, [0] * 5))
            master_model_copy = deepcopy(self.master_model_instance)
            logging.getLogger("pyomo.core").setLevel(logging.ERROR)
            master_model_copy.delta.store_values(open_switches)  # type: ignore

            results = self.master_solver.solve(master_model_copy, tee=False)

            if results.solver.termination_condition != pyo.TerminationCondition.optimal:
                master_feasible.append(False)
                slave_results.append(None)
                continue

            master_feasible.append(True)
            master_ds = master_model_copy.d.extract_values()  # type: ignore
            self.slave_model_instance.master_d.store_values(master_ds)  # type: ignore
            try:
                results = self.slave_solver.solve(self.slave_model_instance, tee=False)
            except Exception as e:
                slave_results.append(None)
                continue
            if results.solver.termination_condition != pyo.TerminationCondition.optimal:
                slave_results.append(None)
            else:
                slave_results.append(self.slave_model_instance.objective())  # type: ignore

        self.switch_combination = switch_combination.with_columns(
            pl.Series(master_feasible).alias("master_feasible"),
            pl.Series(slave_results).alias("slave_results"),
        ).filter(c("master_feasible") == True)

    def test_one_switch_combinations(self, open_switches_list: list[int]) -> None:

        open_switches = dict(zip(open_switches_list, [0] * len(open_switches_list)))
        master_model_copy = deepcopy(self.master_model_instance)
        logging.getLogger("pyomo.core").setLevel(logging.ERROR)
        master_model_copy.delta.store_values(open_switches)  # type: ignore

        results = self.master_solver.solve(master_model_copy, tee=False)

        if results.solver.termination_condition != pyo.TerminationCondition.optimal:
            log.error("Master model is infeasible with the given switch combination.")
            return None

        master_ds = master_model_copy.d.extract_values()  # type: ignore
        self.slave_model_instance.master_d.store_values(master_ds)  # type: ignore
        try:
            results = self.slave_solver.solve(
                self.slave_model_instance, tee=self.verbose
            )
        except Exception as e:
            log.error(f"Slave model failed to solve: {e}")
            return None
        if results.solver.termination_condition != pyo.TerminationCondition.optimal:
            log.error("Slave model is infeasible with the given switch combination.")
            return None

        self.slack_i_sq = extract_optimization_results(
            self.slave_model_instance, "slack_i_sq"
        ).filter(c("slack_i_sq") > self.slack_threshold)

        self.slack_v_pos = extract_optimization_results(
            self.slave_model_instance, "slack_v_pos"
        ).filter(c("slack_v_pos") > self.slack_threshold)

        self.slack_v_neg = extract_optimization_results(
            self.slave_model_instance, "slack_v_neg"
        ).filter(c("slack_v_neg") > self.slack_threshold)
