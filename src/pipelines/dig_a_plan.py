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
from typing import TypedDict, Unpack, Literal


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
from optimization_model.master_model.parameters import master_model_parameters
from optimization_model.master_model.variables import master_model_variables
from optimization_model.master_model.constraints import master_model_constraints

from optimization_model.slave_model.sets import slave_model_sets
from optimization_model.slave_model.parameters import slave_model_parameters
from optimization_model.slave_model.variables import (
    slave_model_variables,
    infeasible_slave_model_variables,
)
from optimization_model.slave_model.constraints import (
    optimal_slave_model_constraints,
    infeasible_slave_model_constraints,
)

from pyomo_utility import extract_optimization_results

log = generate_log(name=__name__)


class DataSchemaPolarsModel(TypedDict, total=True):
    node_data: pl.DataFrame
    edge_data: pl.DataFrame


def generate_master_model() -> pyo.AbstractModel:
    master_model: pyo.AbstractModel = pyo.AbstractModel()  # type: ignore
    master_model = master_model_sets(master_model)
    master_model = master_model_parameters(master_model)
    master_model = master_model_variables(master_model)
    master_model = master_model_constraints(master_model)
    return master_model


def generate_optimal_slave_model() -> pyo.AbstractModel:
    slave_model: pyo.AbstractModel = pyo.AbstractModel()  # type: ignore
    slave_model = slave_model_sets(slave_model)
    slave_model = slave_model_parameters(slave_model)
    slave_model = slave_model_variables(slave_model)
    slave_model = optimal_slave_model_constraints(slave_model)
    slave_model.dual = Suffix(direction=Suffix.IMPORT)
    return slave_model


def generate_infeasible_slave_model() -> pyo.AbstractModel:
    slave_model: pyo.AbstractModel = pyo.AbstractModel()  # type: ignore
    slave_model = slave_model_sets(slave_model)
    slave_model = slave_model_parameters(slave_model)
    slave_model = infeasible_slave_model_variables(slave_model)
    slave_model = infeasible_slave_model_constraints(slave_model)
    slave_model.dual = Suffix(direction=Suffix.IMPORT)
    return slave_model


class DigAPlan:
    def __init__(
        self,
        verbose: bool = False,
        big_m: float = 1e4,
        slack_threshold: float = 1e-5,
        convergence_threshold=1e-4,
        power_factor: float = 1.0,
        current_factor: float = 1.0,
        voltage_factor: float = 1.0,
        slave_objective_type: Literal["losses", "line_loading"] = "losses",
        master_relaxed: bool = False,
    ) -> None:

        self.verbose: int = verbose
        self.big_m: float = big_m
        self.convergence_threshold: float = convergence_threshold
        self.slack_threshold: float = slack_threshold
        self.d: pl.DataFrame = pl.DataFrame()
        self.infeasible_slave: bool
        self.slave_obj: float
        self.master_obj: float = -1e8
        self.power_factor: float = power_factor
        self.current_factor: float = current_factor
        self.voltage_factor: float = voltage_factor

        self.master_obj_list = []
        self.slave_obj_list = []

        self.__node_data: pt.DataFrame[NodeData] = NodeData.DataFrame(
            schema=NodeData.columns
        ).cast()
        self.__edge_data: pt.DataFrame[EdgeData] = EdgeData.DataFrame(
            schema=EdgeData.columns
        ).cast()
        self.__delta_variable: pl.DataFrame

        self.__master_model: pyo.AbstractModel = generate_master_model()
        self.__optimal_slave_model: pyo.AbstractModel = generate_optimal_slave_model()
        self.__infeasible_slave_model: pyo.AbstractModel = (
            generate_infeasible_slave_model()
        )
        self.__master_model_instance: pyo.ConcreteModel
        self.__optimal_slave_model_instance: pyo.ConcreteModel
        self.__infeasible_slave_model_instance: pyo.ConcreteModel

        self.__scaled_optimal_slave_model_instance: pyo.ConcreteModel
        self.__scaled_infeasible_slave_model_instance: pyo.ConcreteModel

        self.__slack_node: int
        self.master_solver = pyo.SolverFactory("gurobi")
        self.master_solver.options["IntegralityFocus"] = (
            1  # To insure master binary variable remains binary
        )
        self.slave_solver = pyo.SolverFactory("gurobi")
        # self.slave_solver.options['NonConvex'] = 2
        # self.slave_solver.options['QCPDual'] = 1
        # # self.slave_solver.options['BarQCPConvTol'] = 1e-5
        # self.slave_solver.options['BarHomogeneous'] = 1

        self.slack_i_sq: pl.DataFrame
        self.slack_v_pos: pl.DataFrame
        self.slack_v_neg: pl.DataFrame
        self.marginal_cost: pl.DataFrame

        # These lists will be updated by the optimization and read by Dash
        self.master_obj_list = []
        self.slave_obj_list = []
        self.convergence_list = []

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
    def optimal_slave_model(self) -> pyo.AbstractModel:
        return self.__optimal_slave_model

    @property
    def infeasible_slave_model(self) -> pyo.AbstractModel:
        return self.__infeasible_slave_model

    @property
    def master_model_instance(self) -> pyo.ConcreteModel:
        return self.__master_model_instance

    @property
    def optimal_slave_model_instance(self) -> pyo.ConcreteModel:
        return self.__optimal_slave_model_instance

    @property
    def infeasible_slave_model_instance(self) -> pyo.ConcreteModel:
        return self.__infeasible_slave_model_instance

    @property
    def scaled_optimal_slave_model_instance(self) -> pyo.ConcreteModel:
        return self.__scaled_optimal_slave_model_instance

    @property
    def scaled_infeasible_slave_model_instance(self) -> pyo.ConcreteModel:
        return self.__scaled_infeasible_slave_model_instance

    @property
    def slack_node(self) -> int:
        return self.__slack_node

    @property
    def delta_variable(self) -> pl.DataFrame:
        return self.__delta_variable

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
                "C": {
                    None: self.edge_data.select(
                        pl.concat_list("edge_id", "u_of_edge", "v_of_edge")
                        .map_elements(tuple, return_dtype=pl.Object)
                        .alias("C")
                    )["C"].to_list()
                    + self.edge_data.select(
                        pl.concat_list("edge_id", "v_of_edge", "u_of_edge")
                        .map_elements(tuple, return_dtype=pl.Object)
                        .alias("C")
                    )["C"].to_list()
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
            }
        }

        self.__master_model_instance = self.master_model.create_instance(grid_data)  # type: ignore
        self.__optimal_slave_model_instance = self.optimal_slave_model.create_instance(grid_data)  # type: ignore
        self.__infeasible_slave_model_instance = self.infeasible_slave_model.create_instance(grid_data)  # type: ignore

        self.__scale_slave_models()

        self.__delta_variable = pl.DataFrame(
            self.master_model_instance.delta.items(),  # type: ignore
            schema=["S", "delta_variable"],
        )

    def __scale_slave_models(self) -> None:

        for model in [
            self.optimal_slave_model_instance,
            self.infeasible_slave_model_instance,
        ]:
            model.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
            model.scaling_factor[model.p_flow] = self.power_factor
            model.scaling_factor[model.q_flow] = self.power_factor
            model.scaling_factor[model.i_sq] = self.current_factor
            model.scaling_factor[model.v_sq] = self.voltage_factor

        self.__scaled_optimal_slave_model_instance = pyo.TransformationFactory(
            "core.scale_model"
        ).create_using(
            self.optimal_slave_model_instance
        )  # type: ignore
        self.__scaled_infeasible_slave_model_instance = pyo.TransformationFactory(
            "core.scale_model"
        ).create_using(
            self.infeasible_slave_model_instance
        )  # type: ignore

    def solve_models_pipeline(self, max_iters: int) -> None:
        convergence_result = np.inf
        master_delta = self.find_initial_state_of_switches()

        pbar = tqdm.tqdm(
            range(max_iters),
            desc="Master obj: 0, Slave obj: 0 and Gap: 1e6",
            disable=False,
        )
        logging.getLogger("pyomo.core").setLevel(logging.ERROR)
        for k in pbar:
            try:
                self.optimal_slave_model_instance.master_delta.store_values(master_delta)  # type: ignore
                self.scaled_optimal_slave_model_instance.master_delta.store_values(master_delta)  # type: ignore
                results = self.slave_solver.solve(
                    self.scaled_optimal_slave_model_instance, tee=self.verbose
                )

                if (
                    results.solver.termination_condition
                    == pyo.TerminationCondition.optimal
                ):
                    pyo.TransformationFactory("core.scale_model").propagate_solution(  # type: ignore
                        self.scaled_optimal_slave_model_instance,
                        self.optimal_slave_model_instance,
                    )
                    self.slave_obj = self.optimal_slave_model_instance.objective()  # type: ignore
                    self.infeasible_slave = False
                else:
                    self.infeasible_slave_model_instance.master_delta.store_values(master_delta)  # type: ignore
                    self.scaled_infeasible_slave_model_instance.master_delta.store_values(master_delta)  # type: ignore
                    _ = self.slave_solver.solve(
                        self.scaled_infeasible_slave_model_instance, tee=self.verbose
                    )
                    pyo.TransformationFactory("core.scale_model").propagate_solution(  # type: ignore
                        self.scaled_infeasible_slave_model_instance,
                        self.infeasible_slave_model_instance,
                    )
                    self.slave_obj = self.scaled_infeasible_slave_model_instance.objective()  # type: ignore
                    self.check_slave_feasibility()
                    self.infeasible_slave = True
            except Exception as e:
                log.error(f"Slave model did not converge: {e}")
                break

            self.add_benders_cut()

            results = self.master_solver.solve(self.master_model_instance, tee=False)
            if results.solver.termination_condition != pyo.TerminationCondition.optimal:
                pbar.disable = True
                log.warning(
                    f"\nMaster model did not converge: {results.solver.termination_condition}"
                )
                break

            self.master_obj = self.master_model_instance.objective()  # type: ignore

            if self.infeasible_slave == False:
                convergence_result = self.slave_obj - self.master_obj  # type: ignore

            pbar.set_description(
                f"Master obj: {self.master_obj:.2E}, Slave obj: {self.slave_obj:.2E} and Gap: {convergence_result:.2E}"
            )

            self.master_obj_list.append(self.master_obj)
            self.slave_obj_list.append(self.slave_obj)
            self.convergence_list.append(self.slave_obj - self.master_obj)  # type: ignore
            if convergence_result < self.convergence_threshold:
                break
            master_delta = self.master_model_instance.delta.extract_values()  # type: ignore

    def add_benders_cut(self) -> None:
        constraint_name = ["master_switch_status_propagation"]
        if self.infeasible_slave == True:

            marginal_cost_df = pl.DataFrame(
                {
                    "name": list(dict(self.infeasible_slave_model_instance.dual).keys()),  # type: ignore
                    "marginal_cost": list(dict(self.infeasible_slave_model_instance.dual).values()),  # type: ignore
                }
            )
            delta_value = extract_optimization_results(
                self.infeasible_slave_model_instance, "master_delta"
            )

        else:
            marginal_cost_df = pl.DataFrame(
                {
                    "name": list(dict(self.optimal_slave_model_instance.dual).keys()),  # type: ignore
                    "marginal_cost": list(dict(self.optimal_slave_model_instance.dual).values()),  # type: ignore
                }
            )

            delta_value = extract_optimization_results(
                self.optimal_slave_model_instance, "master_delta"
            )

        marginal_cost_df = (
            marginal_cost_df.with_columns(
                c("name").map_elements(lambda x: x.name, return_dtype=pl.Utf8),
            )
            .with_columns(
                c("name")
                .pipe(modify_string_col, format_str={"]": ""})
                .str.split("[")
                .list.to_struct(fields=["name", "S"])
            )
            .unnest("name")
            .filter(c("name").is_in(constraint_name))
            .with_columns(c("S").cast(pl.Int64))
        )

        marginal_cost_df = marginal_cost_df.join(delta_value, on="S").join(
            self.delta_variable, on="S"
        )

        self.marginal_cost = marginal_cost_df

        marginal_cost_df = marginal_cost_df.filter(c("master_delta") == 1)

        new_cut = self.slave_obj
        for data in marginal_cost_df.to_dicts():
            new_cut += data["marginal_cost"] * (
                data["delta_variable"] - data["master_delta"]
            )

        if self.infeasible_slave == True:
            self.master_model_instance.infeasibility_cut.add(0 >= new_cut)  # type: ignore
        else:
            self.master_model_instance.optimality_cut.add(self.master_model_instance.theta >= new_cut)  # type: ignore

    def extract_switch_status(self) -> pl.DataFrame:
        switch_status = self.edge_data.filter(c("type") == "switch").with_columns(
            c("edge_id").replace_strict(self.optimal_slave_model_instance.delta.extract_values(), default=None).alias("delta"),  # type: ignore
            (
                ~c("edge_id")
                .replace_strict(self.optimal_slave_model_instance.delta.extract_values(), default=None)  # type: ignore
                .pipe(cast_boolean)
            ).alias("open"),
        )["eq_fk", "edge_id", "delta", "normal_open", "open"]
        return switch_status

    def extract_node_voltage(self) -> pl.DataFrame:
        node_voltage: pl.DataFrame = (
            extract_optimization_results(self.optimal_slave_model_instance, "v_sq")
            .select((c("v_sq")).sqrt().alias("v_pu"), c("N").alias("node_id"))
            .join(
                self.node_data["cn_fk", "node_id", "v_base"], on="node_id", how="left"
            )
        )

        return node_voltage

    def extract_edge_current(self) -> pl.DataFrame:
        edge_current: pl.DataFrame = (
            extract_optimization_results(self.optimal_slave_model_instance, "i_sq")
            .select(
                (c("i_sq")).sqrt().alias("i_pu"), c("C").list.get(0).alias("edge_id")
            )
            .group_by("edge_id")
            .agg(c("i_pu").max())
            .sort("edge_id")
            .join(
                self.edge_data.filter(c("type") != "switch")[
                    "eq_fk", "edge_id", "i_base"
                ],
                on="edge_id",
                how="inner",
            )
        )
        return edge_current

    def check_slave_feasibility(self):
        self.slack_i_sq = extract_optimization_results(
            self.infeasible_slave_model_instance, "slack_i_sq"
        ).filter(c("slack_i_sq") > self.slack_threshold)

        self.slack_v_pos = extract_optimization_results(
            self.infeasible_slave_model_instance, "slack_v_pos"
        ).filter(c("slack_v_pos") > self.slack_threshold)

        self.slack_v_neg = extract_optimization_results(
            self.infeasible_slave_model_instance, "slack_v_neg"
        ).filter(c("slack_v_neg") > self.slack_threshold)

    def find_initial_state_of_switches(self):
        """
        Check if the topology is a tree.
        """
        nx_graph = nx.Graph()
        _ = self.edge_data.filter(~c("normal_open")).with_columns(
            pl.struct(pl.all()).pipe(generate_nx_edge, nx_graph=nx_graph)
        )
        if nx.is_tree(nx_graph):

            slack_node_id = self.node_data.filter(c("type") == "slack")["node_id"][0]
            digraph = generate_bfs_tree_with_edge_data(
                graph=nx_graph, source=slack_node_id
            )
            initial_master_delta = pl_to_dict_with_tuple(
                get_all_edge_data(digraph).select(
                    pl.concat_list("edge_id", "v_of_edge", "u_of_edge").alias("LC"),
                    pl.lit(1).alias("value"),
                )
            )

        else:
            log.warning(
                "The resulting graph considering normal switch status is NOT a tree.\n The initial state of switches is determined solving master model."
            )

            results = self.master_solver.solve(self.master_model_instance, tee=False)
            if results.solver.termination_condition != pyo.TerminationCondition.optimal:
                log.warning(
                    f"\nMaster model did not converge: {results.solver.termination_condition}"
                )

            self.master_obj = self.master_model_instance.objective()  # type: ignore
            initial_master_delta = self.master_model_instance.delta.extract_values()  # type: ignore
        return initial_master_delta
