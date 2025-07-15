import polars as pl
from polars import col as c
import logging
import pyomo.environ as pyo

from pyomo.environ import Suffix
import patito as pt
from typing import TypedDict, Unpack

from general_function import pl_to_dict, generate_log, pl_to_dict_with_tuple
from polars_function import cast_boolean, modify_string_col
from networkx_function import (
    generate_nx_edge,
    generate_bfs_tree_with_edge_data,
    get_all_edge_data,
)

from data_schema.node_data import NodeData
from data_schema.edge_data import EdgeData

from optimization_model.combined_model.sets import model_sets
from optimization_model.combined_model.parameters import model_parameters
from optimization_model.combined_model.variables import model_variables
from optimization_model.combined_model.constraints import combined_model_constraints

from pyomo_utility import extract_optimization_results

log = generate_log(name=__name__)

class DataSchemaPolarsModel(TypedDict, total=True):
    node_data: pl.DataFrame
    edge_data: pl.DataFrame


def generate_combined_model() -> pyo.AbstractModel:
    """Builds the single combined radial + DistFlow model."""
    combined_model: pyo.AbstractModel = pyo.AbstractModel()  # type: ignore
    combined_model = model_sets(combined_model)
    combined_model = model_parameters(combined_model)
    combined_model = model_variables(combined_model)
    combined_model = combined_model_constraints(combined_model)
    return combined_model


class DigAPlan:
    def __init__(
        self,
        verbose: bool = False,
        big_m: float = 1e4,
        power_factor: float = 1.0,
        current_factor: float = 1.0,
        voltage_factor: float = 1.0,
    ) -> None:
        self.verbose = verbose
        self.big_m = big_m
        self.power_factor = power_factor
        self.current_factor = current_factor
        self.voltage_factor = voltage_factor

        # Data storage
        self.__node_data: pt.DataFrame[NodeData] = NodeData.DataFrame(schema=NodeData.columns).cast()
        self.__edge_data: pt.DataFrame[EdgeData] = EdgeData.DataFrame(schema=EdgeData.columns).cast()
        self.__delta_variable: pl.DataFrame
        self.__slack_node: int

        # Build the combined model
        self.__combined_model: pyo.AbstractModel = generate_combined_model()
        self.__combined_model_instance: pyo.ConcreteModel

        # Solver
        self.solver = pyo.SolverFactory("gurobi")
        self.solver.options["IntegralityFocus"] = 1

        # Results: record objective history
        self.combined_obj_list: list[float] = []

    @property
    def node_data(self) -> pt.DataFrame[NodeData]:
        return self.__node_data

    @property
    def edge_data(self) -> pt.DataFrame[EdgeData]:
        return self.__edge_data

    @property
    def slack_node(self) -> int:
        return self.__slack_node

    @property
    def delta_variable(self) -> pl.DataFrame:
        return self.__delta_variable

    @property
    def combined_model_instance(self) -> pyo.ConcreteModel:
        return self.__combined_model_instance

    def add_grid_data(self, **grid_data: Unpack[DataSchemaPolarsModel]) -> None:
        """Load node and edge tables and instantiate the model."""
        for table_name, pl_table in grid_data.items():
            if table_name == "node_data":
                self.__node_data_setter(node_data=pl_table)  # type: ignore
            elif table_name == "edge_data":
                self.__edge_data_setter(edge_data=pl_table)  # type: ignore
            else:
                raise ValueError(f"{table_name} is not a valid name")

        if self.__node_data.filter(c("type") == "slack").height != 1:
            raise ValueError("There must be only one slack node")

        self.__slack_node = self.__node_data.filter(c("type") == "slack")["node_id"][0]
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
        self.__combined_model_instance = self.__combined_model.create_instance(grid_data)  # type: ignore
        self.__delta_variable = pl.DataFrame(
            self.__combined_model_instance.delta.items(),schema=["S", "delta_variable"],) # type: ignore
        

    def solve_combined_model(self) -> None:
        """Solve the combined radial+DistFlow model."""
        results = self.solver.solve(self.__combined_model_instance, tee=self.verbose)
        if results.solver.termination_condition != pyo.TerminationCondition.optimal:
            log.error(f"Solve failed: {results.solver.termination_condition}")
            return
        current_obj = pyo.value(self.__combined_model_instance.objective)
        self.combined_obj_list.append(current_obj) # type: ignore
        log.info(f"Combined solve successful: objective = {current_obj:.4f}")
        self.__delta_variable = pl.DataFrame(
            self.__combined_model_instance.delta.items(), # type: ignore
            schema=["S", "delta_variable"],
        )
        

    # Alias for backward compatibility
    solve = solve_combined_model

    def extract_switch_status(self) -> pl.DataFrame:
        # Build switch status: delta ∈ {0,1}, open when delta == 0
        df = (
            self.__edge_data
            .filter(c("type") == "switch")
            .with_columns(
                # Map each switch edge_id to its delta (default 0 if missing)
                c("edge_id")
                    .replace_strict(
                        self.__combined_model_instance.delta.extract_values(),
                        default=0
                    )
                    .cast(pl.Int64)
                    .alias("delta"),
                # 'open' is True precisely when delta == 0
                (c("delta") == 0).alias("open"),
            )
        )
        # Return only the desired columns
        return df.select(["eq_fk", "edge_id", "delta", "normal_open", "open"])

    def extract_node_voltage(self) -> pl.DataFrame:
        return (
            extract_optimization_results(self.__combined_model_instance, "v_sq")
            .select((c("v_sq")).sqrt().alias("v_pu"), c("N").alias("node_id"))
            .join(self.__node_data[["cn_fk", "node_id", "v_base"]], on="node_id", how="left")
        )

    def extract_edge_current(self) -> pl.DataFrame:
        return (
            extract_optimization_results(self.__combined_model_instance, "i_sq")
            .select((c("i_sq")).sqrt().alias("i_pu"), c("C").list.get(0).alias("edge_id"))
            .group_by("edge_id").agg(c("i_pu").max())
            .sort("edge_id")
            .join(
                self.__edge_data.filter(c("type") != "switch")[["eq_fk", "edge_id", "i_base"]],
                on="edge_id", how="inner"
            )
        )
