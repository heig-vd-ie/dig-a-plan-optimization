import polars as pl
from polars import col as c
import os
os.environ["GRB_LICENSE_FILE"] = os.path.join(os.environ["HOME"], "gurobi_license", "gurobi.lic")

import copy

import pyomo.environ as pyo

from pyomo.environ import Suffix
import patito as pt
from typing import TypedDict, Unpack

from general_function import pl_to_dict, generate_log, pl_to_dict_with_tuple
from polars_function import list_to_list_of_tuple, cast_boolean, modify_string_col

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
    # slave_model.fix_d = pyo.ConstraintList()
    return slave_model

class DigAPlan():
    def __init__(
        self, verbose: bool = False, big_m: float = 1e4, penalty_cost: float = 1e3,
        slack_threshold: float = 1e-5) -> None:
    
        
        self.verbose: int = verbose
        self.big_m: float = big_m
        self.slack_threshold: float = slack_threshold
        self.penalty_cost: float = penalty_cost
        self.marginal_cost: pl.DataFrame = pl.DataFrame()
        self.marginal_cost_fun: pl.DataFrame = pl.DataFrame()
        self.d: pl.DataFrame = pl.DataFrame()
        
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
        self.slave_solver.options["QCPDual"] = 1

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
                "penalty_cost": {None: self.penalty_cost},
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

    def solve_models_pipeline(self, max_iters: int, bender_cut_factor: float = 1e0) -> None:

        for k in range(max_iters):
            print(f"\n--- BENDERS ITERATION {k} ---")
            

            master_results = self.master_solver.solve(self.master_model_instance, tee=self.verbose)
            print(" Master solve status:", master_results.solver.termination_condition)
            print("Master objective:", self.master_model_instance.objective()) # type: ignore
            master_ds = self.master_model_instance.d.extract_values() # type: ignore
            # This is needed to avoid infeasibility in the slave model
            master_ds  = dict(map(lambda x: (x[0], 0 if x[1] < 1e-1 else 1), master_ds.items()))
            master_ds_df = pl.DataFrame({
                    "LC": list(master_ds.keys()),
                    str(k): list(master_ds.values())
                }).with_columns(
                    c("LC").cast(pl.List(pl.Utf8)).list.join(",").alias("LC")
                ).sort("LC")
            if self.d.is_empty():
                self.d = master_ds_df
            else:
                self.d = self.d.join(master_ds_df, on="LC", how="inner")
            
            self.slave_model_instance.master_d.store_values(master_ds) # type: ignore
            self.master_model_instance.previous_d.store_values(master_ds) # type: ignore
            
            slave_results = self.slave_solver.solve(self.slave_model_instance, tee=self.verbose)
            print("Slave objective:", self.slave_model_instance.objective()) # type: ignore
        
            
            self.infeasible_i_sq = extract_optimization_results(self.slave_model_instance, "slack_i_sq")\
                .filter(c("slack_i_sq") > self.slack_threshold)
            # self.infeasible_v_sq = extract_optimization_results(self.slave_model_instance, "slack_v_sq")\
            #     .filter(c("slack_v_sq") > self.slack_threshold)
            
            # infeas_pos = (
            #     extract_optimization_results(self.slave_model_instance, "slack_v_pos")
            #     .filter(c("slack_v_pos") > self.slack_threshold)
            #     .with_columns(pl.lit("pos").alias("dir"))
            # )
            # # grab all nodes that violated the lower bound
            # infeas_neg = (
            #         extract_optimization_results(self.slave_model_instance, "slack_v_neg")
            #         .filter(c("slack_v_neg") > self.slack_threshold)
            #         .with_columns(pl.lit("neg").alias("dir"))
            # )
            # # union them into one table
            # self.infeasible_v_sq = pl.concat([infeas_pos, infeas_neg], how="vertical")
            print(self.infeasible_i_sq)
            # if self.infeasible_i_sq.is_empty() & self.infeasible_v_sq.is_empty():
            #     log.info(f"Master model solved in {k} iterations")
            #     break
            # else:
            #     if k <= max_iters:
            #         self.calculate_marginal_costs()
            
            self.master_model_instance.previous_d.store_values(master_ds) # type: ignore
            self.master_model_instance.slave_objective.store_values({None: self.slave_model_instance.objective()}) # type: ignore
            self.add_benders_cut(nb_iter = k, bender_cut_factor = bender_cut_factor)
            self.calculate_marginal_costs()
                    

    def calculate_marginal_costs(self) -> None:
        self.marginal_cost_fun = pl.DataFrame({
                "name": list(dict(self.slave_model_instance.dual).keys()), # type: ignore
                "marginal_cost": list(dict(self.slave_model_instance.dual).values()) # type: ignore
            }).with_columns(
                c("name").map_elements(lambda x: x.name, return_dtype=pl.Utf8),
                pl.when(c("marginal_cost").abs() <= 1e-8).then(pl.lit(0.0)).otherwise(c("marginal_cost")).alias("marginal_cost")
            ).with_columns(
                c("name").pipe(modify_string_col, format_str= {"]":""}).str.split("[").list.to_struct(fields=["name", "index"])
            ).unnest("name")\
            .with_columns(
                c("index").str.split(",").cast(pl.List(pl.Int32)).list.to_struct(fields=["l", "i", "j"])
            ).unnest("index")
    
    def add_benders_cut(self, nb_iter: int, bender_cut_factor = 1e0) -> None:
        constraint_dict = {
            "neg_node_active_power_balance": 1, 
            "pos_node_active_power_balance": 1,
            "neg_node_reactive_power_balance": 1, 
            "pos_node_reactive_power_balance": 1,
            "voltage_drop_lower": 1,
            "voltage_drop_upper": 1,
        }
        
        
        
        marginal_cost_df = pl.DataFrame({
                "name": list(dict(self.slave_model_instance.dual).keys()), # type: ignore
                "marginal_cost": list(dict(self.slave_model_instance.dual).values()) # type: ignore
            }).with_columns(
                c("name").map_elements(lambda x: x.name, return_dtype=pl.Utf8),
                pl.when(c("marginal_cost").abs() <= 1e-8).then(pl.lit(0.0)).otherwise(c("marginal_cost")).alias("marginal_cost")
            ).with_columns(
                c("name").pipe(modify_string_col, format_str= {"]":""}).str.split("[").list.to_struct(fields=["name", "index"])
            ).unnest("name").filter(c("name").is_in(constraint_dict.keys()))\
            .with_columns(
                (c("marginal_cost") * c("name").replace_strict(constraint_dict, default=None)).alias("marginal_cost"),
                c("index").str.split(",").cast(pl.List(pl.Int32)).list.to_struct(fields=["l", "i", "j"])
            ).unnest("index")\
            .group_by(["l", "i", "j"]).agg(c("marginal_cost").sum()).sort(["l", "i", "j"])\
            .with_columns(pl.concat_list(["l", "i", "j"]).cast(pl.List(pl.Utf8)).list.join(",").alias("LC"))
        if self.marginal_cost.is_empty():
            self.marginal_cost = marginal_cost_df.select("LC", c("marginal_cost").alias(str(nb_iter)))
        else:
            self.marginal_cost = self.marginal_cost.join(
                marginal_cost_df.select("LC", c("marginal_cost").alias(str(nb_iter))), on="LC", how="inner")
        
            
        self.master_model_instance.marginal_cost.store_values(  # type: ignore
            pl_to_dict_with_tuple(marginal_cost_df.select(pl.concat_list(["l", "i", "j"]), "marginal_cost")))
            
        # Extract d results from master model
        # master_d_results = extract_optimization_results(self.master_model_instance, "d")\
        #     .with_columns(
        #         c("LC").cast(pl.List(pl.Utf8)).list.join(",").alias("LC")
        #     )
        
            
    # def add_benders_cut(self, nb_iter: int, bender_cut_factor = 1e0) -> None:
    #     constraint_name_list = [
    #         "node_active_power_balance", 
    #         "node_reactive_power_balance",
    #         "voltage_drop_lower",
    #         "voltage_drop_upper",
    #     ]
        
    #     # Extract d results from master model
    #     master_d_results = extract_optimization_results(self.master_model_instance, "d")\
    #         .with_columns(
    #             pl.when(c("d").abs() == 0).then(pl.lit(-1.0)).otherwise(pl.lit(1.0)).alias("factor"),
    #             c("LC").cast(pl.List(pl.Utf8)).list.join(",").alias("LC")
    #         )
    #     # Extract d variable from master model
    #     master_d_variable = pl.DataFrame(
    #         self.master_model_instance.d.items(), # type: ignore
    #         schema=["LC", "d_variable"]
    #         ).with_columns(
    #             c("LC").cast(pl.List(pl.Utf8)).list.join(",").alias("LC")
    #         )
    #     # Extract optimality_cuts from slave model
    #     optimality_cuts = self.marginal_cost.filter(c("name").is_in(constraint_name_list))\
    #         .group_by(["l", "i", "j"]).agg(c("marginal_cost").sum()).sort(["l", "i", "j"])\
    #         .with_columns(
    #             pl.concat_list(["l", "i", "j"]).cast(pl.List(pl.Utf8)).list.join(",").alias("LC")
    #         )
    #     # Join optimality_cuts with master_d_results and master_d_variable
    #     optimality_cuts = optimality_cuts\
    #         .join(master_d_results, on="LC", how="inner")\
    #         .join(master_d_variable, on="LC", how="inner")\
    #         .filter(c("marginal_cost") != 0.0).sort("l")
        
    #     index_list = self.infeasible_i_sq.with_columns(
    #         c("LC").cast(pl.List(pl.Utf8)).list.join(",")
    #     )["LC"].to_list()

    #     feasibility_cuts = self.marginal_cost.filter(c("name") == "current_limit").with_columns(
    #         pl.concat_list("l", "i", "j").cast(pl.List(pl.Utf8)).list.join(",").alias("LC")
    #     ).filter(c("LC").is_in(index_list))
        
    #     feasibility_cuts = feasibility_cuts\
    #         .join(master_d_results, on="LC", how="inner")\
    #         .join(master_d_variable, on="LC", how="inner")\
    #         .filter(c("marginal_cost") != 0.0).sort("l")
    #     # Generate bender_cuts expression 
    #     theta = 0
    #     for data in optimality_cuts.to_dicts():
    #         theta += data["marginal_cost"] * data["factor"] * (data["d"] - data["d_variable"])
        
    #     for data in feasibility_cuts.to_dicts():
    #         theta += data["marginal_cost"] * data["factor"] * (data["d"] - data["d_variable"])
        
    #     # Add Bender cut to master model
    #     self.master_model_instance.add_component(f'theta_{nb_iter}', pyo.Var())
    #     self.master_model_instance.add_component(
    #         f'bender_cuts_{nb_iter}', 
    #         pyo.Constraint(expr=getattr(self.master_model_instance, f'theta_{nb_iter}')>= theta)
    #         )
        
    #     # update objective function
    #     self.master_model_instance.objective.set_value( # type: ignore
    #         expr=self.master_model_instance.objective.expr + # type: ignore
    #         bender_cut_factor* getattr(self.master_model_instance, f'theta_{nb_iter}')
    #         ) 
        
    # def check_all_fd_duals(self, epsilon: float = 1e-6):
    #     """
    #     For every (l,i,j) in current_rotated_cone:
    #     1) Solve slave to get objective1(f1) and dual mu.
    #     2) Deep-copy and loosen that one SOC constraint rhs += epsilon.
    #     3) Resolve to get objective2(f2), compute FD = (f2 - f1)/epsilon.
    #     4) Report mu, FD, and relative error.
    #     """
    #     base = self.__slave_model_instance
    #     # 1) Baseline solve
    #     self.slave_solver.solve(base)
    #     f1 = pyo.value(base.objective)
    # def check_all_fd_duals(self, epsilon: float = 1e-6):
    #     """
    #     For every (l,i,j) in current_rotated_cone:
    #     1) Solve slave to get objective1(f1) and dual mu.
    #     2) Deep-copy and loosen that one SOC constraint rhs += epsilon.
    #     3) Resolve to get objective2(f2), compute FD = (f2 - f1)/epsilon.
    #     4) Report mu, FD, and relative error.
    #     """
    #     base = self.__slave_model_instance
    #     # 1) Baseline solve
    #     self.slave_solver.solve(base)
    #     f1 = pyo.value(base.objective)

    #     results = []
    #     for idx in base.current_rotated_cone:
    #         # 1a) read the dual on this SOC constraint
    #         mu = base.dual[ base.current_rotated_cone[idx] ]
    #     results = []
    #     for idx in base.current_rotated_cone:
    #         # 1a) read the dual on this SOC constraint
    #         mu = base.dual[ base.current_rotated_cone[idx] ]

    #         # 2) perturb only this constraint
    #         pert = copy.deepcopy(base)
    #         con: pyo.Constraint = pert.current_rotated_cone[idx]
    #         # original: con.body <= con.upper()
    #         orig_rhs = con.upper()
    #         # loosen the RHS by +epsilon
    #         con.set_value(expr=(con.body <= orig_rhs + epsilon))
    #         # 2) perturb only this constraint
    #         pert = copy.deepcopy(base)
    #         con: pyo.Constraint = pert.current_rotated_cone[idx]
    #         # original: con.body <= con.upper()
    #         orig_rhs = con.upper()
    #         # loosen the RHS by +epsilon
    #         con.set_value(expr=(con.body <= orig_rhs + epsilon))

    #         # 3) re-solve perturbed
    #         self.slave_solver.solve(pert)
    #         f2 = pyo.value(pert.objective)
    #         # 3) re-solve perturbed
    #         self.slave_solver.solve(pert)
    #         f2 = pyo.value(pert.objective)

    #         # 4) finite-difference
    #         fd = (f2 - f1)#/epsilon
    #         rel_err = ((fd - mu)/mu) if mu != 0 else None
    #         # 4) finite-difference
    #         fd = (f2 - f1)#/epsilon
    #         rel_err = ((fd - mu)/mu) if mu != 0 else None

    #         results.append({
    #             "l,i,j": idx,
    #             "dual (μ)": mu,
    #             "FD approx": fd,
    #             "rel error": rel_err,
    #         })
    #         results.append({
    #             "l,i,j": idx,
    #             "dual (μ)": mu,
    #             "FD approx": fd,
    #             "rel error": rel_err,
    #         })

    #     # 5) print a simple table
    #     print(f"{'idx':>12} | {'dual':>10} | {'FD':>10} | {'rel err':>8}")
    #     print("-"*50)
    #     for r in results:
    #         idx, mu, fd= r["l,i,j"], r["dual (μ)"], r["FD approx"],# r["rel error"]
    #         print(f"{str(idx):>12} | {mu:10.4e} | {fd:10.4e} | ") #
    #         # f"{(f'{re:.2%}' if re is not None else '  n/a'):>8}")
    #     # 5) print a simple table
    #     print(f"{'idx':>12} | {'dual':>10} | {'FD':>10} | {'rel err':>8}")
    #     print("-"*50)
    #     for r in results:
    #         idx, mu, fd= r["l,i,j"], r["dual (μ)"], r["FD approx"],# r["rel error"]
    #         print(f"{str(idx):>12} | {mu:10.4e} | {fd:10.4e} | ") #
    #         # f"{(f'{re:.2%}' if re is not None else '  n/a'):>8}")

    #     return results
    #     return results
    
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