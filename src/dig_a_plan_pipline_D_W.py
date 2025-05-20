# src/dig_a_plan_pipeline_dw.py

import os
import networkx as nx
import pyomo.environ as pyo
from pyomo.environ import Suffix, SolverFactory
import polars as pl
from polars import col as c
from typing import TypedDict, Dict, Any, List, Tuple

from general_function import pl_to_dict, pl_to_dict_with_tuple
from polars_function import list_to_list_of_tuple

# D–W master pieces
from D_W_model.sets        import D_W_model_sets
from D_W_model.parameters  import D_W_model_parameters
from D_W_model.variables   import D_W_model_variables
from D_W_model.constraints import D_W_model_constraints

# DistFlow slave pieces
from slave_model.sets        import slave_model_sets
from slave_model.parameters  import slave_model_parameters
from slave_model.variables   import slave_model_variables
from slave_model.constraints import slave_model_constraints

class DataSchemaPolarsModel(TypedDict, total=True):
    node_data: pl.DataFrame
    edge_data: pl.DataFrame

class DigAPlan:
    def __init__(
        self,
        solver_name: str = "gurobi",
        big_m: float = 1e4,
        v_penalty_cost: float = 1e-3,
        i_penalty_cost: float = 1e-3,
        slack_threshold: float = 1e-5,
        verbose: bool = False,
    ):
        self.verbose         = verbose
        self.big_m           = big_m
        self.v_penalty_cost  = v_penalty_cost
        self.i_penalty_cost  = i_penalty_cost
        self.slack_threshold = slack_threshold

        # Polars frames
        self._node_df: pl.DataFrame = pl.DataFrame()
        self._edge_df: pl.DataFrame = pl.DataFrame()
        self._slack_node: int = -1

        # Build D–W master abstract
        self._master = pyo.AbstractModel()
        D_W_model_sets(self._master)
        D_W_model_parameters(self._master)
        D_W_model_variables(self._master)
        D_W_model_constraints(self._master)

        # Build DistFlow slave abstract
        self._slave = pyo.AbstractModel()
        slave_model_sets(self._slave)
        slave_model_parameters(self._slave)
        slave_model_variables(self._slave)
        slave_model_constraints(self._slave)
        self._slave.dual = Suffix(direction=Suffix.IMPORT)

        # Solver factories
        os.environ["GRB_LICENSE_FILE"] = os.path.join(
            os.environ.get("HOME", "."), "gurobi_license", "gurobi.lic"
        )
        self.master_solver = SolverFactory(solver_name)
        self.slave_solver  = SolverFactory(solver_name)
        self.slave_solver.options["NonConvex"] = 2
        self.slave_solver.options["QCPDual"]   = 1

    def _node_data_setter(self, node_data: pl.DataFrame):
        self._node_df = node_data

    def _edge_data_setter(self, edge_data: pl.DataFrame):
        self._edge_df = edge_data

    def _instantiate_model(self):
        # Collect lists of ids
        node_ids = self._node_df["node_id"].to_list()
        edge_ids = self._edge_df["edge_id"].to_list()
        switch_ids = self._edge_df.filter(c("type") == "switch")["edge_id"].to_list()
    
        # Parameters as dicts
        r = pl_to_dict(self._edge_df["edge_id", "r_pu"])
        x = pl_to_dict(self._edge_df["edge_id", "x_pu"])
        b = pl_to_dict(self._edge_df["edge_id", "b_pu"])
        i_max = pl_to_dict(self._edge_df["edge_id", "i_max_pu"])
        p_node = pl_to_dict(self._node_df["node_id", "p_node_pu"])
        q_node = pl_to_dict(self._node_df["node_id", "q_node_pu"])
        v_min = pl_to_dict(self._node_df["node_id", "v_min_pu"])
        v_max = pl_to_dict(self._node_df["node_id", "v_max_pu"])

        # n_transfo needs a tuple key, eg (edge_id, u_of_edge, v_of_edge)
        n_transfo = pl_to_dict_with_tuple(
            self._edge_df.select(
                pl.concat_list("edge_id", "u_of_edge", "v_of_edge"),
                "n_transfo"
            )
        )
        # C: for each edge, list of (u, v)
        C = pl_to_dict(
            self._edge_df.select(
                "edge_id",
                pl.concat_list("u_of_edge", "v_of_edge").pipe(list_to_list_of_tuple)
            )
        )

        # Build grid_data for Pyomo
        grid_data = {
            "N": node_ids,                 # Set of nodes
            "L": edge_ids,                 # Set of edges
            "S": switch_ids,               # Set of switches
            "C": C,                        # Dict: {edge_id: [(i, j)]}
            "r": r,                        # Dict: {edge_id: value}
            "x": x,
            "b": b,
            "n_transfo": n_transfo,        # Dict: {(edge_id, i, j): value}
            "p_node": p_node,              # Dict: {node_id: value}
            "q_node": q_node,
            "i_max": i_max,                # Dict: {edge_id: value}
            "v_min": v_min,                # Dict: {node_id: value}
            "v_max": v_max,                # Dict: {node_id: value}
            "slack_node": self._slack_node,                        # Scalar
            "slack_node_v_sq": float(self._node_df.filter(c("type") == "slack")["v_node_sqr_pu"][0]),  # Scalar
            "big_m": self.big_m,
            "v_penalty_cost": self.v_penalty_cost,
            "i_penalty_cost": self.i_penalty_cost,
        }
        if self.verbose:
            print("=== grid_data keys ===", list(grid_data.keys()))
            print("N (nodes):", grid_data["N"])
            print("L (edges):", grid_data["L"])
            print("C (first 2):", {k: grid_data["C"][k] for k in list(grid_data["C"])[:2]})

        self._master_inst = self._master.create_instance(grid_data)
        self._master_inst.dual = Suffix(direction=Suffix.IMPORT)
        self._slave_inst = self._slave.create_instance(grid_data)
        self._slave_inst.dual = Suffix(direction=Suffix.IMPORT)


    def add_grid_data(self, **grid_data: Dict[str, pl.DataFrame]) -> None:
        # Extract node_data and edge_data from grid_data dict
        node_data = grid_data.get("node_data")
        edge_data = grid_data.get("edge_data")

        if node_data is None or edge_data is None:
            raise ValueError("Both 'node_data' and 'edge_data' must be provided.")

        # Ensure v_min_pu and v_max_pu columns exist
        nd = node_data
        if "v_min_pu" not in nd.columns:
            nd = nd.with_columns(pl.lit(0.95).alias("v_min_pu"))
        if "v_max_pu" not in nd.columns:
            nd = nd.with_columns(pl.lit(1.05).alias("v_max_pu"))

        # Use internal setters (these will set self._node_df and self._edge_df)
        self._node_data_setter(nd)
        self._edge_data_setter(edge_data)

        # Validate slack node
        if self._node_df.filter(c("type") == "slack").height != 1:
            raise ValueError("There must be only one slack node")
        self._slack_node = int(self._node_df.filter(c("type") == "slack")["node_id"][0])

        # Instantiate models
        self._instantiate_model()


    def solve_column_generation(self, max_iters: int = 20) -> List[Dict[str, Any]]:
        columns: List[Dict[str,Any]] = []
        # 1) seed with an initial pattern
        d0 = self._initial_tree_pattern()
        print("\n==== Initial Tree Pattern Debug ====")
        for (l, i, j), on in d0.items():
            if on:
                print(f"Branch {l} ({i}--{j}) is ON")
        print("Slack node:", self._slack_node)
        print("Nodes with demand:")
        for n, p in self._slave_inst.p_node.items():
            if pyo.value(p) != 0.0:
                print(f"  Node {n}: p = {pyo.value(p)}, q = {pyo.value(self._slave_inst.q_node[n])}")
        f0 = self._solve_slave(d0)
        columns.append({"d": d0, "cost": f0})
        for it in range(max_iters):
            m = self._master_inst
            # 1) Load the current columns into the Param/K set
            m.K.clear()
            for k in range(len(columns)):
                m.K.add(k)
            # 2) Update column_cost & column_d parameters
            for k in m.K:
                m.column_cost[k] = columns[k]["cost"]
                for (l,i,j), bit in columns[k]["d"].items():
                    m.column_d[k, l, i, j] = bit
            # 3) Now that m.K is nonempty, rebuild the D–W constraints
            m.convexity.clear()
            m.convexity.add(sum(m.lambda_k[k] for k in m.K) == 1)
            m.coupling.clear()
            for (l,i,j) in m.LC:
                m.coupling.add(
                    sum(m.column_d[k, l, i, j] * m.lambda_k[k] for k in m.K)
                    == m.delta[l]
                )
            # 4) Solve the restricted master
            self.master_solver.solve(m, tee=self.verbose)
            z = pyo.value(m.objective)
            if self.verbose:
                print(f"Master LP objective = {z:.6f}")
            # 5) Extract the convexity dual (always at index 1 of the list)
            sigma = m.dual[m.convexity[1]]
            # 6) Extract coupling duals in the same order as LC
            pi = {
                (l,i,j): m.dual[m.coupling[idx+1]]
                for idx, (l,i,j) in enumerate(m.LC)
            }
            # 7) Pricing: find new tree and solve slave
            d_star, f_star, rc = self._price_and_solve(pi, sigma)
            if self.verbose:
                print(f" New reduced cost = {rc:.6f}, slave cost = {f_star:.6f}")
            # 8) Convergence check
            if rc >= -1e-6:
                if self.verbose:
                    print("Column generation converged.")
                break
            # 9) Otherwise add the new column and repeat
            columns.append({"d": d_star, "cost": f_star})
        return columns

    def _initial_tree_pattern(self) -> Dict[Tuple[Any,Any,Any], int]:
        G = nx.MultiGraph()  # Use MultiGraph to allow keys
        for (l,i,j) in self._master_inst.LC:
            G.add_edge(i, j, key=l)
        T = nx.minimum_spanning_tree(G)
        return {
            (l,i,j): 1 if T.has_edge(i, j, key=l) else 0
            for (l,i,j) in self._master_inst.LC
        }

    def _solve_slave(self, pattern):
        s = self._slave_inst
        # Set pattern
        for (l, i, j), bit in pattern.items():
            s.master_d[l, i, j] = bit
        results = self.slave_solver.solve(s, tee=self.verbose)
        print(results)
        print("=== SLAVE MODEL SOLUTION VARIABLES ===")
        s.pprint()
        print("\n==== Solution Debug ====")
        for n in s.N:
            try:
                print(f"Node {n}: p_node = {pyo.value(s.p_node[n])}, v_sq = {pyo.value(s.v_sq[n])}")
            except Exception as e:
                print(f"Node {n}: Error accessing variable: {e}")
        for (l, i, j) in s.LC:
            try:
                print(f"Line {l} ({i}->{j}): i_sq = {pyo.value(s.i_sq[l, i, j])}")
            except Exception as e:
                print(f"Line {l} ({i}->{j}): Error accessing variable: {e}")
        for v in s.component_objects(pyo.Var, active=True):
            print("Variable", v)
            varobject = getattr(s, str(v))
            for index in varobject:
                print("   ", index, pyo.value(varobject[index]))
        return pyo.value(s.objective)

    def _price_and_solve(
        self,
        pi: Dict[Tuple[Any,Any,Any], float],
        sigma: float
    ) -> Tuple[Dict[Tuple,Any], float, float]:
        G = nx.MultiGraph()
        for (l,i,j) in self._master_inst.LC:
            w = -(pi.get((l,i,j),0.0) + pi.get((l,j,i),0.0)) / 2
            G.add_edge(i, j, key=l, weight=w)
        T = nx.minimum_spanning_tree(G)
        d_star = {
            (l,i,j): 1 if T.has_edge(i, j, key=l) else 0
            for (l,i,j) in self._master_inst.LC
        }
        f_star = self._solve_slave(d_star)
        rc = f_star - sum(pi[e] * d_star[e] for e in d_star) - sigma
        return d_star, f_star, rc
