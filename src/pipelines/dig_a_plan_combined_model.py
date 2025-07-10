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
from networkx_function import generate_nx_edge, generate_bfs_tree_with_edge_data, get_all_edge_data

from data_schema.node_data import NodeData
from data_schema.edge_data import EdgeData

from optimization_model.combined_model.sets import model_sets
from optimization_model.combined_model.parameters import model_parameters
from optimization_model.combined_model.variables import model_variables
from optimization_model.combined_model.constraints import model_constraints



from pyomo_utility import extract_optimization_results

log = generate_log(name=__name__)

class DataSchemaPolarsModel(TypedDict, total=True):
    node_data: pl.DataFrame
    edge_data: pl.DataFrame

def generate_model() -> pyo.AbstractModel:
    model: pyo.AbstractModel = pyo.AbstractModel() # type: ignore
    model = model_sets(model)
    model = model_parameters(model)
    model = model_variables(model)
    model = model_constraints(model)
    return model



class DigAPlan():
    def __init__(
        self, verbose: bool = False, big_m: float = 1e4, penalty_cost: float = 1e3, current_factor: float = 1e2,
        voltage_factor: float = 1e1, power_factor: float = 1e1,
        infeasibility_factor: float = 4, optimality_factor: float = 1,
        slack_threshold: float = 1e-5, convergence_threshold=1e-4) -> None:
    
        self.verbose: int = verbose
        self.big_m: float = big_m
        self.convergence_threshold: float = convergence_threshold
        self.slack_threshold: float = slack_threshold
        self.d: pl.DataFrame = pl.DataFrame()
        self.infeasible_slave: bool
        self.model_obj: float
        self.penalty_cost: float = penalty_cost
        self.current_factor: float = current_factor
        self.voltage_factor: float = voltage_factor
        self.power_factor: float = power_factor
        self.infeasibility_factor: float = infeasibility_factor
        self.optimality_factor: float = optimality_factor
        
        self.__node_data: pt.DataFrame[NodeData] = NodeData.DataFrame(schema=NodeData.columns).cast()
        self.__edge_data: pt.DataFrame[EdgeData] = EdgeData.DataFrame(schema=EdgeData.columns).cast()
        self.__model: pyo.AbstractModel = generate_model()
        self.__model_instance: pyo.ConcreteModel

        

        self.__slack_node : int
        self.solver = pyo.SolverFactory('gurobi')
        self.solver.options['IntegralityFocus'] = 1 # To insure master binary variable remains binary
        # self.solver.options['NumericFocus'] = 3 # To insure master binary variable remains binary
        # self.solver.options['TimeLimit'] = 60 # To insure master binary variable remains binary
        
        self.slack_i_sq: pl.DataFrame    
        self.slack_v_pos: pl.DataFrame
        self.slack_v_neg: pl.DataFrame
        self.marginal_cost: pl.DataFrame

    @property
    def node_data(self) -> pt.DataFrame[NodeData]:
        return self.__node_data
    @property
    def edge_data(self) -> pt.DataFrame[EdgeData]:
        return self.__edge_data
    @property
    def model(self) -> pyo.AbstractModel:
        return self.__model
    @property
    def model_instance(self) -> pyo.ConcreteModel:
        return self.__model_instance

    @property
    def slack_node(self) -> int:
        return self.__slack_node

    def __node_data_setter(self, node_data: pl.DataFrame):
        old_table: pl.DataFrame = self.__node_data.clear()
        col_list: list[str] = list(set(node_data.columns).intersection(set(old_table.columns)))
        new_table_pl: pl.DataFrame = pl.concat([old_table, node_data.select(col_list)], how="diagonal_relaxed")
        new_table_pt: pt.DataFrame[NodeData] = pt.DataFrame(new_table_pl)\
            .set_model(NodeData).fill_null(strategy="defaults").cast(strict=True)
        new_table_pt.validate()
        self.__node_data = new_table_pt
    
    def __edge_data_setter(self, edge_data: pl.DataFrame):
        old_table: pl.DataFrame = self.__edge_data.clear()
        col_list: list[str] = list(set(edge_data.columns).intersection(set(old_table.columns)))
        new_table_pl: pl.DataFrame = pl.concat([old_table, edge_data.select(col_list)], how="diagonal_relaxed")
        new_table_pt: pt.DataFrame[EdgeData] = pt.DataFrame(new_table_pl)\
            .set_model(EdgeData).fill_null(strategy="defaults").cast(strict=True)
        new_table_pt.validate()
        self.__edge_data = new_table_pt
        
    def __instantiate_model(self):
        
        grid_data = {
            None: {
                "N": {None: self.node_data["node_id"].to_list()},
                "L": {None: self.edge_data["edge_id"].to_list()},
                "C": dict(zip(
                                    self.edge_data["edge_id"].to_list(),
                                    map(lambda x: [tuple(x)],
                                        self.edge_data.select(pl.concat_list("u_of_edge", "v_of_edge").alias("nodes"))["nodes"].to_list()
                                    )
                                )),
                "S": {None: self.edge_data.filter(c("type") == "switch")["edge_id"].to_list()},
                "r": pl_to_dict(self.edge_data["edge_id", "r_pu"]),
                "x": pl_to_dict(self.edge_data["edge_id", "x_pu"]),
                "b": pl_to_dict(self.edge_data["edge_id", "b_pu"]),
                "n_transfo": pl_to_dict_with_tuple(
                    self.edge_data.select(pl.concat_list("edge_id", "u_of_edge", "v_of_edge"), "n_transfo")),
                "p_node": pl_to_dict(self.node_data["node_id", "p_node_pu"]),
                "q_node": pl_to_dict(self.node_data["node_id", "q_node_pu"]),
                "i_max": pl_to_dict(self.edge_data["edge_id", "i_max_pu"]),
                "v_min": pl_to_dict(self.node_data["node_id", "v_min_pu"]),
                "v_max": pl_to_dict(self.node_data["node_id", "v_max_pu"]),
                "slack_node": {None: self.slack_node},
                "slack_node_v_sq": {None: self.node_data.filter(c("type") == "slack")["v_node_sqr_pu"][0]},
                "big_m": {None: self.big_m},
                "penalty_cost": {None: self.penalty_cost},
                "current_factor": {None: self.current_factor},
                "voltage_factor": {None: self.voltage_factor},
                "power_factor": {None: self.power_factor},
            }
        }
        
        self.__model_instance = self.model.create_instance(grid_data) # type: ignore
        
    def add_grid_data(self, **grid_data: Unpack[DataSchemaPolarsModel]) -> None:
        
        for table_name, pl_table in grid_data.items():
            if table_name  == "node_data":
                self.__node_data_setter(node_data = pl_table) # type: ignore
            elif table_name  == "edge_data":
                self.__edge_data_setter(edge_data = pl_table) # type: ignore
            else:
                raise ValueError(f"{table_name} is not a valid name")
        
        if self.node_data.filter(c("type") == "slack").height != 1:
            raise ValueError("There must be only one slack node")
        
        self.__slack_node: int = self.node_data.filter(c("type") == "slack")["node_id"][0]
        self.__instantiate_model()
        self.__adapt_penalty_cost()
        
    
    def __adapt_penalty_cost(self):
        """
        Returns the penalty factor used in the optimization.
        """
        self.penalty_cost = 10**round(log10(abs(self.edge_data["r_pu"].max()))) * self.penalty_cost # type: ignore

    def solve_models_pipeline(self) -> None:
        
        logging.getLogger('pyomo.core').setLevel(logging.ERROR)
        with tqdm.tqdm(total=1, desc="Solving model") as pbar:
            _ = self.solver.solve(self.__model_instance, tee=self.verbose)
            self.check_slave_feasibility()
            self.model_obj = self.__model_instance.objective() # type: ignore
            pbar.update()


    def extract_switch_status(self) -> pl.DataFrame:
        switch_status = self.edge_data.filter(c("type") == "switch")\
            .with_columns(
            (~c("edge_id").replace_strict(self.model_instance.delta.extract_values(), default=None) # type: ignore
            .pipe(cast_boolean)
            ).alias("open")
        )["eq_fk", "edge_id","normal_open", "open"]
        return switch_status
    
    def extract_node_voltage(self) -> pl.DataFrame:
        node_voltage: pl.DataFrame = extract_optimization_results(self.model_instance, "v_sq")\
        .select(
            (self.voltage_factor*c("v_sq")).sqrt().alias("v_pu"),
            c("N").alias("node_id")
        ).join(
            self.node_data["cn_fk", "node_id", "v_base"], on="node_id", how="left"
        )
        
        return node_voltage
    
    def extract_edge_current(self) -> pl.DataFrame:
        edge_current: pl.DataFrame = extract_optimization_results(self.model_instance, "i_sq")\
            .select(
                (self.current_factor*c("i_sq")).sqrt().alias("i_pu"),
                c("LC").list.get(0).alias("edge_id")
            ).group_by("edge_id").agg(c("i_pu").max()).sort("edge_id")\
            .join(
                self.edge_data.filter(c("type") != "switch")["eq_fk", "edge_id", "i_base"], on="edge_id", how="inner"
            )
        return edge_current
    
    def check_slave_feasibility(self):
        self.slack_i_sq = extract_optimization_results(self.model_instance, "slack_i_sq")\
                .filter(c("slack_i_sq") > self.slack_threshold)
                
        self.slack_v_pos = extract_optimization_results(self.model_instance, "slack_v_pos")\
            .filter(c("slack_v_pos") > self.slack_threshold)
            
        self.slack_v_neg = extract_optimization_results(self.model_instance, "slack_v_neg")\
            .filter(c("slack_v_neg") > self.slack_threshold)
    