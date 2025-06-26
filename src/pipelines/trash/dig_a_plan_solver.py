import polars as pl
from polars import col as c
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

from data_schema.node_data import NodeData
from data_schema.edge_data import EdgeData

# Master model building functions
from optimization_model.master_model.sets import master_model_sets
from optimization_model.master_model.parameters import master_model_parameters
from optimization_model.master_model.variables import master_model_variables
from optimization_model.master_model.constraints import master_model_constraints

# Slave model building functions
from optimization_model.slave_model.sets import slave_model_sets
from optimization_model.slave_model.parameters import slave_model_parameters
from optimization_model.slave_model.variables import slave_model_variables, slack_variables
from optimization_model.slave_model.constraints import (
    feasible_slave_model_constraints,
    infeasible_slave_model_constraints
)

from pyomo_utility import extract_optimization_results

log = generate_log(name=__name__)

class DataSchemaPolarsModel(TypedDict, total=True):
    node_data: pl.DataFrame
    edge_data: pl.DataFrame

# Generate abstract models

def generate_master_model() -> pyo.AbstractModel:
    m = pyo.AbstractModel()
    m = master_model_sets(m)
    m = master_model_parameters(m)
    m = master_model_variables(m)
    m = master_model_constraints(m)
    return m


def generate_feasible_slave_model() -> pyo.AbstractModel:
    m = pyo.AbstractModel()
    m = slave_model_sets(m)
    m = slave_model_parameters(m)
    m = slave_model_variables(m)
    m = feasible_slave_model_constraints(m)
    return m


def generate_infeasible_slave_model() -> pyo.AbstractModel:
    m = pyo.AbstractModel()
    m = slave_model_sets(m)
    m = slave_model_parameters(m)
    m = slave_model_variables(m)
    m = slack_variables(m)
    m = infeasible_slave_model_constraints(m)
    return m

class DigAPlan:
    def __init__(
        self,
        verbose: bool = False,
        big_m: float = 1e4,
        penalty_cost: float = 1e2,
        slack_threshold: float = 1e-5,
        convergence_threshold: float = 1e-4
    ) -> None:
        # Settings
        self.verbose = verbose
        self.big_m = big_m
        self.penalty_cost = penalty_cost
        self.slack_threshold = slack_threshold
        self.convergence_threshold = convergence_threshold

        # Dataframes
        self.__node_data: pt.DataFrame[NodeData] = NodeData.DataFrame(schema=NodeData.columns).cast()
        self.__edge_data: pt.DataFrame[EdgeData] = EdgeData.DataFrame(schema=EdgeData.columns).cast()

        # Abstract models
        self.__master_model = generate_master_model()
        self.__slave_model = generate_feasible_slave_model()
        self.__infeasible_slave_model = generate_infeasible_slave_model()

        # ConcreteModel instances (initialized later)
        self.__master_model_instance: pyo.ConcreteModel
        self.__slave_model_instance: pyo.ConcreteModel
        self.__infeasible_slave_model_instance: pyo.ConcreteModel

        # Solvers
        self.master_solver = pyo.SolverFactory('gurobi')
        self.master_solver.options['IntegralityFocus'] = 1

        self.slave_solver = pyo.SolverFactory('gurobi_persistent')
        self.slave_solver.options.update({
            'NonConvex':      2,
            'QCPDual':        1,
            'Method':         2,
            'DualReductions': 0,
        })

        self.infeasible_slave_solver = pyo.SolverFactory('gurobi_persistent')
        self.infeasible_slave_solver.options.update({
            'NonConvex':      2,
            'QCPDual':        1,
            'Method':         2,
            'DualReductions': 0,
            'BarQCPConvTol':  1e-8,
            'NumericFocus':   3,
            'FeasibilityTol': 1e-9,
            'OptimalityTol':  1e-9,
            'ScaleFlag':      3,
        })

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
    def infeasible_slave_model(self) -> pyo.AbstractModel:
        return self.__infeasible_slave_model

    @property
    def master_model_instance(self) -> pyo.ConcreteModel:
        return self.__master_model_instance

    @property
    def slave_model_instance(self) -> pyo.ConcreteModel:
        return self.__slave_model_instance

    @property
    def infeasible_slave_model_instance(self) -> pyo.ConcreteModel:
        return self.__infeasible_slave_model_instance

    @property
    def slack_node(self) -> int:
        return self.__slack_node

    def __node_data_setter(self, node_data: pl.DataFrame):
        # 1) Ensure all six bound columns exist
        required = ["p_node_min_pu","p_node_max_pu",
                    "q_node_min_pu","q_node_max_pu",
                    "v_min_pu","v_max_pu"]
        for col in required:
            if col not in node_data.columns:
                node_data = node_data.with_columns([pl.lit(None).alias(col)])

        # 2) Fill nulls so that p and q are fixed, and voltage gets soft bounds
        node_data = node_data.with_columns([
            pl.col("p_node_min_pu")
            .fill_null(pl.col("p_node_pu"))
            .alias("p_node_min_pu"),
            pl.col("p_node_max_pu")
            .fill_null(pl.col("p_node_pu"))
            .alias("p_node_max_pu"),
            pl.col("q_node_min_pu")
            .fill_null(pl.col("q_node_pu"))
            .alias("q_node_min_pu"),
            pl.col("q_node_max_pu")
            .fill_null(pl.col("q_node_pu"))
            .alias("q_node_max_pu"),
            pl.col("v_min_pu")
            .fill_null(0.95)
            .alias("v_min_pu"),
            pl.col("v_max_pu")
            .fill_null(1.05)
            .alias("v_max_pu"),
        ])

        # 3) Merge, cast and validate as before…
        old = self.__node_data.clear()
        cols = list(set(node_data.columns) & set(old.columns))
        df = pl.concat([old, node_data.select(cols)], how="diagonal_relaxed")
        new_pt = (
            pt.DataFrame(df)
            .set_model(NodeData)
            .fill_null(strategy="defaults")
            .cast(strict=True)
        )
        new_pt.validate()
        self.__node_data = new_pt




    def __edge_data_setter(self, edge_data: pl.DataFrame):
        # fill nulls before casting
        edge_data = edge_data.with_columns([
            pl.col('r_pu').fill_null(1e-6),
            pl.col('x_pu').fill_null(1e-6),
            pl.col('g_pu').fill_null(0.0),
            pl.col('b_pu').fill_null(0.0),
            pl.col('n_transfo').fill_null(1.0),
        ])

        old = self.__edge_data.clear()
        cols = list(set(edge_data.columns).intersection(set(old.columns)))
        df = pl.concat([old, edge_data.select(cols)], how='diagonal_relaxed')
        new_pt = pt.DataFrame(df)\
            .set_model(EdgeData)\
            .fill_null(strategy="defaults")\
            .cast(strict=True)
        new_pt.validate()
        self.__edge_data = new_pt


    def __instantiate_model(self):
        # prepare data dict
        data = {None: {
            'N': {None: self.node_data['node_id'].to_list()},
            'L': {None: self.edge_data['edge_id'].to_list()},
            'C': pl_to_dict(self.edge_data.select('edge_id', pl.concat_list('u_of_edge','v_of_edge').pipe(list_to_list_of_tuple))),
            'S': {None: self.edge_data.filter(c('type')=='switch')['edge_id'].to_list()},
            'r': pl_to_dict(self.edge_data['edge_id','r_pu']),
            'x': pl_to_dict(self.edge_data['edge_id','x_pu']),
            'b': pl_to_dict(self.edge_data['edge_id','b_pu']),
            'n_transfo': pl_to_dict_with_tuple(self.edge_data.select(pl.concat_list('edge_id','u_of_edge','v_of_edge'),'n_transfo')),
            'p_node': pl_to_dict(self.node_data['node_id','p_node_pu']),
            'q_node': pl_to_dict(self.node_data['node_id','q_node_pu']),
            'i_max': pl_to_dict(self.edge_data['edge_id','i_max_pu']),
            'v_min': pl_to_dict(self.node_data['node_id','v_min_pu']),
            'v_max': pl_to_dict(self.node_data['node_id','v_max_pu']),
            'slack_node': {None: self.slack_node},
            'slack_node_v_sq': {None: self.node_data.filter(c('type')=='slack')['v_node_sqr_pu'][0]},
            'big_m': {None: self.big_m},
            'penalty_cost': {None: self.penalty_cost},
        }}
        # create instances
        self.__master_model_instance = self.master_model.create_instance(data)
        self.__slave_model_instance = self.slave_model.create_instance(data)
        self.__infeasible_slave_model_instance = self.infeasible_slave_model.create_instance(data)
        # attach dual suffix to slave instances
        self.__slave_model_instance.dual = Suffix(direction=Suffix.IMPORT)
        self.__infeasible_slave_model_instance.dual = Suffix(direction=Suffix.IMPORT)
        # register with persistent solvers
        self.slave_solver.set_instance(self.__slave_model_instance)
        self.infeasible_slave_solver.set_instance(self.__infeasible_slave_model_instance)

    def add_grid_data(self, **grid_data: Unpack[DataSchemaPolarsModel]) -> None:
        for name, df in grid_data.items():
            if name=='node_data': self.__node_data_setter(df)
            elif name=='edge_data': self.__edge_data_setter(df)
            else: raise ValueError(f'Unknown table {name}')
        slack = self.__node_data.filter(c('type')=='slack')
        if slack.height!=1: raise ValueError('Require exactly one slack node')
        self.__slack_node = int(slack['node_id'][0])
        self.__instantiate_model()
        # adapt penalty if needed
        self.penalty_cost = 10**round(log10(abs(self.edge_data['r_pu'].max()))) * self.penalty_cost

    def solve_models_pipeline(self, max_iters: int):
        pbar = tqdm.tqdm(range(max_iters), desc='Master obj: ...', disable=False)
        logging.getLogger('pyomo.core').setLevel(logging.ERROR)
        for k in pbar:
            # master solve
            master_res = self.master_solver.solve(self.__master_model_instance, tee=self.verbose)
            if master_res.solver.termination_condition!=pyo.TerminationCondition.optimal:
                log.warning(f'Master did not converge: {master_res.solver.termination_condition}')
                break
            self.master_obj = pyo.value(self.__master_model_instance.objective)
            # transfer d
            ds = self.__master_model_instance.d.extract_values()
            self.__slave_model_instance.master_d.store_values(ds)
            # slave solve
            slave_res = self.slave_solver.solve(tee=self.verbose)
            if slave_res.solver.termination_condition!=pyo.TerminationCondition.optimal:
                # infeasible slave
                self.infeasible_slave = True
                convergence = np.inf
                self.__infeasible_slave_model_instance.master_d.store_values(ds)
                res2 = self.infeasible_slave_solver.solve(tee=self.verbose)
                self.slave_obj = pyo.value(self.__infeasible_slave_model_instance.objective)
            else:
                self.infeasible_slave = False
                self.slave_obj = pyo.value(self.__slave_model_instance.objective)
                losses = self.__master_model_instance.losses.extract_values()[None]
                convergence = self.slave_obj - self.master_obj + losses
            pbar.set_description(f'Master obj: {self.master_obj:.3E}, Slave obj: {self.slave_obj:.3E}, Gap: {convergence:.3E}')
            if convergence<self.convergence_threshold:
                break
            self.add_benders_cut()
        
    
    def add_benders_cut(self) -> None:
        if self.infeasible_slave == True:
            constraint_dict = {
                    "node_active_power_balance": 1,
                    "node_reactive_power_balance": 1,
                    "voltage_drop_lower": 1,
                    "voltage_drop_upper": 1,
                    # "current_limit": 1,
                    # "voltage_upper_limits": 1,
                    # "voltage_lower_limits": 1,
                }
            marginal_cost_df = pl.DataFrame({
                "name": list(dict(self.infeasible_slave_model_instance.dual).keys()), # type: ignore
                "marginal_cost": list(dict(self.infeasible_slave_model_instance.dual).values()) # type: ignore
            })

        else:
            constraint_dict = {
                    "current_rotated_cone":-1,
                }

            marginal_cost_df = pl.DataFrame({
                "name": list(dict(self.slave_model_instance.dual).keys()), # type: ignore
                "marginal_cost": list(dict(self.slave_model_instance.dual).values()) # type: ignore
            })

        # Extract delta results from master model
        d_value = extract_optimization_results(self.master_model_instance, "d").select(
                c("LC").cast(pl.List(pl.Utf8)).list.join(",").alias("LC"), "d"
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
            .group_by("LC").agg(c("marginal_cost").sum())
    

        marginal_cost_df = marginal_cost_df\
            .join(d_value, on="LC")\
            .join(d_variable, on="LC")\

        new_cut = self.slave_obj 
        for data in marginal_cost_df.to_dicts():
            new_cut += data["marginal_cost"] * (data["d"] - data["d_variable"]) 

        if self.infeasible_slave == True:  
            self.master_model_instance.infeasibility_cut.add(0  >= new_cut) # type: ignore
        else:   
            self.master_model_instance.optimality_cut.add(self.master_model_instance.theta >= new_cut) # type: ignore

        
    def extract_switch_status(self) -> dict[str, bool]:
        switch_status = self.master_model_instance.delta.extract_values() # type: ignore
        return pl_to_dict(
            self.edge_data.filter(c("type") == "switch")
                .with_columns(
                    c("edge_id").replace_strict(switch_status, default=None).pipe(cast_boolean).alias("closed")
            )["eq_fk", "closed"]
        )
    
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