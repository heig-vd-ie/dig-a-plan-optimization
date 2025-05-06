import polars as pl
from polars import col as c
import os
os.environ["GRB_LICENSE_FILE"] = os.path.join(os.environ["HOME"], "gurobi_license", "gurobi.lic")

import pyomo.environ as pyo

from pyomo.environ import Suffix
import patito as pt
from typing import TypedDict, Unpack

from general_function import pl_to_dict, generate_log, pl_to_dict_with_tuple
from polars_function import list_to_list_of_tuple, cast_boolean

from data_schema.node_data import NodeData
from data_schema.edge_data import EdgeData

from master_model.sets import master_model_sets
from master_model.parameters import master_model_parameters
from master_model.variables import master_model_variables
from master_model.constraints import master_model_constraints

from slave_model.sets import slave_model_sets
from slave_model.parameters import slave_model_parameters
from slave_model.variables import slave_model_variables
from slave_model.constraints import slave_model_constraints

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

def generate_slave_model() -> pyo.AbstractModel:
    slave_model: pyo.AbstractModel = pyo.AbstractModel() # type: ignore
    slave_model = slave_model_sets(slave_model)
    slave_model = slave_model_parameters(slave_model)
    slave_model = slave_model_variables(slave_model)
    slave_model = slave_model_constraints(slave_model)
    slave_model.dual = Suffix(direction=Suffix.IMPORT)
    #add fix_d list
    slave_model.fix_d = pyo.ConstraintList()
    return slave_model

class DigAPlan():
    def __init__(self, verbose: bool = False, big_m: float = 1e4, v_penalty_cost: float = 1e-3, i_penalty_cost: float = 1e-3):
        
        self.verbose: int = verbose
        self.big_m: float = big_m
        self.v_penalty_cost: float = v_penalty_cost
        self.i_penalty_cost: float = i_penalty_cost
        
        self.__node_data: pt.DataFrame[NodeData] = NodeData.DataFrame(schema=NodeData.columns).cast()
        self.__edge_data: pt.DataFrame[EdgeData] = EdgeData.DataFrame(schema=EdgeData.columns).cast()
        self.__master_model: pyo.AbstractModel = generate_master_model()
        self.__slave_model: pyo.AbstractModel = generate_slave_model()
        self.__master_model_instance: pyo.ConcreteModel
        self.__slave_model_instance: pyo.ConcreteModel
        self.__slack_node : int
        self.master_solver = pyo.SolverFactory('gurobi')
        self.slave_solver = pyo.SolverFactory('gurobi')
        self.slave_solver.options['NonConvex'] = 2

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
                "v_penalty_cost": {None: self.v_penalty_cost},
                "i_penalty_cost": {None: self.i_penalty_cost},
            }
        }
        
        self.__master_model_instance = self.master_model.create_instance(grid_data) # type: ignore
        self.__slave_model_instance = self.slave_model.create_instance(grid_data) # type: ignore
        
        # Attach dual suffix to slave model instance for shadow prices
        self.__slave_model_instance.dual = Suffix(direction=Suffix.IMPORT)
        
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
        
    def solve_master_model(self):
        _ = self.master_solver.solve(self.master_model_instance, tee=self.verbose)
        
        
    
    # def solve_slave_model(self): #OLD
    #     new_d = self.master_model_instance.d.extract_values() # type: ignore
    #     self.slave_model_instance.master_d.store_values(new_d) # type: ignore
    #     _ = self.slave_solver.solve(self.slave_model_instance, tee=self.verbose)
        
    def solve_slave_model(self):
        # 1) pull current master d’s
        master_ds = self.master_model_instance.d.extract_values()
        # 2) clear any fix_d constraints
        self.slave_model_instance.fix_d.clear()
        # 3) load them into slave.master_d
        self.slave_model_instance.master_d.store_values(master_ds)
        # 3) re-add one fix_d constraint per (l,i,j)
        for (l,i,j), dval in master_ds.items():
            self.slave_model_instance.fix_d.add(
                self.slave_model_instance.d[l,i,j] == dval
            )

        # 2) solve the slave
        result = self.slave_solver.solve(self.slave_model_instance, tee=self.verbose)

        # 3) scrape off the non‐zero duals from just the fix_d block
        duals = []
        for idx, cons in self.slave_model_instance.fix_d.items():
            sigma = self.slave_model_instance.dual[cons]
            if abs(sigma) > 1e-8:
                duals.append((idx, sigma))

        # 4) return them so the Benders driver can build cuts
        return result, duals

    def solve_models_pipeline(self, max_iters=20, tol=1e-6):
        m = self.master_model_instance
        s = self.slave_model_instance
        solver_m = self.master_solver
        solver_s = self.slave_solver

        prev_theta = None
        for k in range(1, max_iters+1):
            # 1) Solve master
            res_m = solver_m.solve(m, tee=self.verbose)
            assert res_m.solver.termination_condition == pyo.TerminationCondition.optimal
            # slave solve & get duals
            res_s, duals = self.solve_slave_model()
            # if infeasible
            if res_s.solver.termination_condition!=pyo.TerminationCondition.optimal:
                expr = sum(sigma*(m.d[idx]-dval) for idx,sigma in duals for dval in [self.master_model_instance.d[idx].value])
                m.add_component(f'feas_cut_{k}', pyo.Constraint(expr=expr>=0))
                continue
            # feasible: add optimality cut
            w = pyo.value(s.objective)
            expr = sum(-sigma*(m.d[idx]-dval) for idx,sigma in duals for dval in [self.master_model_instance.d[idx].value])
            m.add_component(f'opt_cut_{k}', pyo.Constraint(expr=m.Theta>=w+expr))
            th = pyo.value(m.Theta)
            if prev_theta is not None and abs(th-w)<tol:
                print(f'Benders converged in {k} iters')
                break
            prev_theta = th

            


        
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