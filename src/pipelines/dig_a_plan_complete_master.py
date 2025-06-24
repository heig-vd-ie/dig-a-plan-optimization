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

from optimization_model.master_model.sets import master_model_sets
from optimization_model.master_model.parameters import master_model_parameters
from optimization_model.master_model.variables import master_model_variables
from optimization_model.master_model.constraints import master_model_constraints

from optimization_model.slave_model.sets import slave_model_sets
from optimization_model.slave_model.parameters import slave_model_parameters
from optimization_model.slave_model.variables import slave_model_variables
from optimization_model.slave_model.constraints import slave_model_constraints

from pyomo_utility import extract_optimization_results

log = generate_log(name=__name__)

class DataSchemaPolarsModel(TypedDict, total=True):
    node_data: pl.DataFrame
    edge_data: pl.DataFrame

def generate_master_model() -> pyo.AbstractModel:
    master_model: pyo.AbstractModel = pyo.AbstractModel() # type: ignore
    master_model = master_model_sets(master_model)
    master_model = master_model_parameters(master_model)
    master_model = master_model_variables(master_model)
    master_model = master_model_constraints(master_model)
    return master_model

def generate_feasible_slave_model() -> pyo.AbstractModel:
    slave_model: pyo.AbstractModel = pyo.AbstractModel() # type: ignore
    slave_model = slave_model_sets(slave_model)
    slave_model = slave_model_parameters(slave_model)
    slave_model = slave_model_variables(slave_model)
    slave_model = slave_model_constraints(slave_model)
    slave_model.dual = Suffix(direction=Suffix.IMPORT)
    return slave_model


class DigAPlan():
    def __init__(
        self, verbose: bool = False, big_m: float = 1e4, penalty_cost: float = 1e2,
        slack_threshold: float = 1e-5, convergence_threshold=1e-4) -> None:
    
        self.verbose: int = verbose
        self.big_m: float = big_m
        self.convergence_threshold: float = convergence_threshold
        self.slack_threshold: float = slack_threshold
        self.d: pl.DataFrame = pl.DataFrame()
        self.infeasible_slave: bool
        self.slave_obj: float
        self.master_obj: float = -1e-8
        self.penalty_cost: float = penalty_cost
        
        self.__node_data: pt.DataFrame[NodeData] = NodeData.DataFrame(schema=NodeData.columns).cast()
        self.__edge_data: pt.DataFrame[EdgeData] = EdgeData.DataFrame(schema=EdgeData.columns).cast()
        self.__master_model: pyo.AbstractModel = generate_master_model()
        
        self.__slave_model: pyo.AbstractModel = generate_feasible_slave_model()
        self.__master_model_instance: pyo.ConcreteModel
        self.__slave_model_instance: pyo.ConcreteModel
        

        self.__slack_node : int
        self.master_solver = pyo.SolverFactory('gurobi')
        self.master_solver.options['IntegralityFocus'] = 1 # To insure master binary variable remains binary
        self.slave_solver = pyo.SolverFactory('gurobi')
        # self.slave_solver.options['NonConvex'] = 2 # To allow non-convex optimization
        # self.slave_solver.options["QCPDual"] = 1 # To allow dual variables extraction on quadratic constraints
        # self.slave_solver.options["NumericFocus"] = 1 # To allow dual variables extraction on quadratic constraints
        
        
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
                "C": pl_to_dict(
                    self.edge_data.select("edge_id", pl.concat_list("u_of_edge", "v_of_edge").pipe(list_to_list_of_tuple))),
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
            }
        }
        
        self.__master_model_instance = self.master_model.create_instance(grid_data) # type: ignore
        self.__slave_model_instance = self.slave_model.create_instance(grid_data) # type: ignore
        
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

    def solve_models_pipeline(self, max_iters: int) -> None:
        
        self.find_initial_state_of_switches()
        
        pbar = tqdm.tqdm(range(max_iters), desc="Master obj: 0, Slave obj: 0 and Gap: 1e6", disable = False)
        logging.getLogger('pyomo.core').setLevel(logging.ERROR)
        for k in pbar:

            results = self.slave_solver.solve(self.__slave_model_instance, tee=self.verbose)
            self.slave_obj = self.__slave_model_instance.objective() # type: ignore
            
            self.check_slave_feasibility()
            self.add_benders_cut()
            
            results = self.master_solver.solve(self.master_model_instance, tee=False)
            if results.solver.termination_condition != pyo.TerminationCondition.optimal:
                pbar.disable = True
                log.warning(f"\nMaster model did not converge: {results.solver.termination_condition}")
                break
            
            self.master_obj = self.master_model_instance.objective() # type: ignore
            
            if self.infeasible_slave == True:
                convergence_result = np.inf
            else:
                self.infeasible_slave = False
                convergence_result = (
                    self.slave_obj - self.master_obj # type: ignore
                )
            pbar.set_description(
                f"Master obj: {self.master_obj:.1E}, Slave obj: {self.slave_obj:.1E} and Gap: {convergence_result:.1E}"
                ) 
            
            if convergence_result < self.convergence_threshold:
                break 
        
            master_ds = self.master_model_instance.d.extract_values() # type: ignore
            self.__slave_model_instance.master_d.store_values(master_ds) # type: ignore
        
    
    def add_benders_cut(self) -> None:
        if self.infeasible_slave == True:
            constraint_dict = {
                    "node_active_power_balance": 1,
                    "node_reactive_power_balance": 1,
                    # "voltage_drop_lower": 1,
                    # "voltage_drop_upper": 1,
                    "current_limit": -1,
                    "voltage_upper_limits": 1,
                    "voltage_lower_limits": -1,
                    # "current_rotated_cone": 1,
                }

        else:
            constraint_dict = {
                "node_active_power_balance": 1,
                "node_reactive_power_balance": 1,
                # "voltage_drop_lower": 1,
                # "voltage_drop_upper": 1,
                # "current_limit": 1,
                # "voltage_upper_limits": 1,
                # "voltage_lower_limits": 1,
                "current_rotated_cone": -1,
                }

        marginal_cost_df = pl.DataFrame({
            "name": list(dict(self.slave_model_instance.dual).keys()), # type: ignore
            "marginal_cost": list(dict(self.slave_model_instance.dual).values()) # type: ignore
        })

        # Extract delta results from master model
        d_value = extract_optimization_results(self.slave_model_instance, "master_d").select(
                c("LC").cast(pl.List(pl.Utf8)).list.join(",").alias("LC"), c("master_d").alias("d")
            )

        d_variable = pl.DataFrame(
            self.master_model_instance.d.items(), # type: ignore
            schema=["LC", "d_variable"]
        ).with_columns(
            c("LC").cast(pl.List(pl.Utf8)).list.join(",").alias("LC")
        )

        # Extract marginal costs from slave model    
        marginal_cost_df = marginal_cost_df\
            .with_columns(
                c("name").map_elements(lambda x: x.name, return_dtype=pl.Utf8),
            ).with_columns(
                c("name").pipe(modify_string_col, format_str= {"]":""}).str.split("[").list.to_struct(fields=["name", "index"])
            ).unnest("name").filter(c("name").is_in(constraint_dict.keys())).with_columns(
                (c("marginal_cost") * c("name").replace_strict(constraint_dict, default=None)).alias("marginal_cost"),
                c("index").str.split(",").cast(pl.List(pl.Int32)).list.to_struct(fields=["l", "i", "j"])
            ).unnest("index")\
            .with_columns(
                pl.concat_list("l", "i", "j").cast(pl.List(pl.Utf8)).list.join(",").alias("LC")
            )      
        
        marginal_cost_df: pl.DataFrame = marginal_cost_df\
            .group_by("LC").agg(c("marginal_cost").sum())\
            .filter(c("marginal_cost").abs() >=1e-7)
    
        self.marginal_cost = marginal_cost_df
        
        marginal_cost_df = marginal_cost_df\
            .join(d_value, on="LC")\
            .join(d_variable, on="LC")\

        new_cut = self.slave_obj 
        for data in marginal_cost_df.to_dicts():
            new_cut += data["marginal_cost"] * (data["d"] - data["d_variable"]) 

        if self.infeasible_slave == True:  
            self.master_model_instance.infeasibility_cut.add(0 >= new_cut) # type: ignore
        else:   
            self.master_model_instance.optimality_cut.add(self.master_model_instance.theta >= new_cut) # type: ignore
            

    def extract_switch_status(self) -> pl.DataFrame:
        switch_status = self.edge_data.filter(c("type") == "switch")\
            .with_columns(
            (~c("edge_id").replace_strict(self.master_model_instance.delta.extract_values(), default=None) # type: ignore
            .pipe(cast_boolean)
            ).alias("open")
        )["eq_fk", "edge_id","normal_open", "open"]
        return switch_status
    
    def extract_node_voltage(self) -> pl.DataFrame:
        node_voltage: pl.DataFrame = extract_optimization_results(self.slave_model_instance, "v_sq")\
        .select(
            c("v_sq").sqrt().alias("v_pu"),
            c("N").alias("node_id")
        ).join(
            self.node_data["cn_fk", "node_id", "v_base"], on="node_id", how="left"
        )
        
        return node_voltage
    
    def extract_edge_current(self) -> pl.DataFrame:
        edge_current: pl.DataFrame = extract_optimization_results(self.slave_model_instance, "i_sq")\
            .select(
                c("i_sq").sqrt().alias("i_pu"),
                c("LC").list.get(0).alias("edge_id")
            ).group_by("edge_id").agg(c("i_pu").max()).sort("edge_id")\
            .join(
                self.edge_data.filter(c("type") != "switch")["eq_fk", "edge_id", "i_base"], on="edge_id", how="inner"
            )
        return edge_current
    
    def check_slave_feasibility(self):
        self.slack_i_sq = extract_optimization_results(self.slave_model_instance, "slack_i_sq")\
                .filter(c("slack_i_sq") > self.slack_threshold)
                
        self.slack_v_pos = extract_optimization_results(self.slave_model_instance, "slack_v_pos")\
            .filter(c("slack_v_pos") > self.slack_threshold)
            
        self.slack_v_neg = extract_optimization_results(self.slave_model_instance, "slack_v_neg")\
            .filter(c("slack_v_neg") > self.slack_threshold)

        if (self.slack_i_sq.height > 0) or (self.slack_v_pos.height > 0) or (self.slack_v_neg.height > 0):
            self.infeasible_slave = True
        else:
            self.infeasible_slave = False
    
    def find_initial_state_of_switches(self):
        """
        Check if the topology is a tree.
        """
        nx_graph = nx.Graph()
        _ = self.edge_data.filter(~c("normal_open")).with_columns(
            pl.struct(pl.all()).pipe(generate_nx_edge, nx_graph=nx_graph)
        )
        if nx.is_tree(nx_graph):

            
            slack_node_id = self.node_data.filter(c("type")=="slack")["node_id"][0]
            digraph = generate_bfs_tree_with_edge_data(graph = nx_graph, source=slack_node_id)
            initial_master_d = pl_to_dict_with_tuple(get_all_edge_data(digraph)\
                .select(
                    pl.concat_list("edge_id", "v_of_edge", "u_of_edge").alias("LC"),
                    pl.lit(1).alias("value")
                ))
            
        else:
            log.warning(
                "The resulting graph considering normal switch is NOT a tree.\n The initial state of switches is determined solving master model.")
            
        results = self.master_solver.solve(self.master_model_instance, tee=False)
        if results.solver.termination_condition != pyo.TerminationCondition.optimal:
            log.warning(f"\nMaster model did not converge: {results.solver.termination_condition}")

        self.master_obj = self.master_model_instance.objective() # type: ignore
        initial_master_d = self.master_model_instance.d.extract_values() # type: ignore 
            
        self.slave_model_instance.master_d.store_values(initial_master_d) # type: ignore
