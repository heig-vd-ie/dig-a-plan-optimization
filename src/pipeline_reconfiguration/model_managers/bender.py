import pyomo.environ as pyo
import gurobipy as gp
import tqdm
import polars as pl
import numpy as np
import logging
import networkx as nx
from helpers import pl_to_dict_with_tuple, generate_log
from polars import col as c
from helpers import modify_string_col
from helpers import (
    generate_nx_edge,
    generate_bfs_tree_with_edge_data,
    get_all_edge_data,
)
from pipeline_reconfiguration.data_manager import PipelineDataManager
from pipeline_reconfiguration.configs import BenderConfig, PipelineType
from model_reconfiguration import (
    generate_master_model,
    generate_infeasible_slave_model,
    generate_optimal_slave_model,
)
from helpers.pyomo import extract_optimization_results
from pipeline_reconfiguration.model_managers import PipelineModelManager

log = generate_log(name=__name__)


class PipelineModelManagerBender(PipelineModelManager):

    def __init__(
        self,
        config: BenderConfig,
        data_manager: PipelineDataManager,
        pipeline_type=PipelineType.BENDER,
    ) -> None:
        """Initialize the Bender model manager with configuration and data manager"""
        super().__init__(config, data_manager, pipeline_type)

        self.master_model: pyo.AbstractModel = generate_master_model(
            relaxed=config.master_relaxed
        )
        self.optimal_slave_model: pyo.AbstractModel = generate_optimal_slave_model()
        self.infeasible_slave_model: pyo.AbstractModel = (
            generate_infeasible_slave_model()
        )
        self.master_model_instance: pyo.ConcreteModel
        self.optimal_slave_model_instance: pyo.ConcreteModel
        self.infeasible_slave_model_instance: pyo.ConcreteModel
        self.scaled_optimal_slave_model_instance: pyo.ConcreteModel
        self.scaled_infeasible_slave_model_instance: pyo.ConcreteModel

        self.d: pl.DataFrame = pl.DataFrame()
        self.infeasible_slave: bool = False
        self.slave_obj: float = 0.0
        self.master_obj: float = -1e8

        self.master_obj_list = []
        self.slave_obj_list = []

        self.slack_i_sq: pl.DataFrame = pl.DataFrame()
        self.slack_v_pos: pl.DataFrame = pl.DataFrame()
        self.slack_v_neg: pl.DataFrame = pl.DataFrame()
        self.marginal_cost: pl.DataFrame = pl.DataFrame()

        # These lists will be updated by the optimization and read by Dash
        self.master_obj_list = []
        self.slave_obj_list = []
        self.convergence_list = []

        self.master_solver = pyo.SolverFactory(config.solver_name)
        self.master_solver.options["Seed"] = config.seed
        if config.threads is not None:
            self.master_solver.options["Threads"] = config.threads
        self.master_solver.options["IntegralityFocus"] = (
            config.master_solver_integrality_focus
        )  # To insure master binary variable remains binary
        self.slave_solver = pyo.SolverFactory(config.solver_name)
        self.slave_solver.options["Seed"] = config.seed
        if config.threads is not None:
            self.slave_solver.options["Threads"] = config.threads
        if config.solver_non_convex is not None:
            self.slave_solver.options["NonConvex"] = config.solver_non_convex
        if config.solver_qcp_dual is not None:
            self.slave_solver.options["QCPDual"] = config.solver_qcp_dual
        if config.solver_bar_qcp_conv_tol is not None:
            self.slave_solver.options["BarQCPConvTol"] = config.solver_bar_qcp_conv_tol
        if config.solver_bar_homogeneous is not None:
            self.slave_solver.options["BarHomogeneous"] = config.solver_bar_homogeneous

    def __scale_slave_models(
        self, factor_p: float, factor_q: float, factor_i: float, factor_v: float
    ) -> None:

        for model in [
            self.optimal_slave_model_instance,
            self.infeasible_slave_model_instance,
        ]:
            model.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
            model.scaling_factor[model.p_flow] = factor_p
            model.scaling_factor[model.q_flow] = factor_q
            model.scaling_factor[model.i_sq] = factor_i
            model.scaling_factor[model.v_sq] = factor_v

        self.scaled_optimal_slave_model_instance = pyo.TransformationFactory(
            "core.scale_model"
        ).create_using(  # type: ignore
            self.optimal_slave_model_instance
        )
        self.scaled_infeasible_slave_model_instance = pyo.TransformationFactory(
            "core.scale_model"
        ).create_using(  # type: ignore
            self.infeasible_slave_model_instance
        )

    def instantaniate_model(self, grid_data_parameters_dict: dict | None) -> None:
        grid_data_parameters_dict = grid_data_parameters_dict[list(grid_data_parameters_dict.keys())[0]]  # type: ignore
        self.master_model_instance = self.master_model.create_instance(grid_data_parameters_dict)  # type: ignore
        self.optimal_slave_model_instance = self.optimal_slave_model.create_instance(grid_data_parameters_dict)  # type: ignore
        self.infeasible_slave_model_instance = self.infeasible_slave_model.create_instance(grid_data_parameters_dict)  # type: ignore

        self.δ_variable = pl.DataFrame(
            self.master_model_instance.δ.items(),  # type: ignore
            schema=["S", "δ_variable"],
        )
        self.ζ_variable = pl.DataFrame(
            self.master_model_instance.ζ.items(),  # type: ignore
            schema=["TrTaps", "ζ_variable"],
        )
        self.__scale_slave_models(
            factor_p=self.config.factor_p,
            factor_q=self.config.factor_q,
            factor_i=self.config.factor_i,
            factor_v=self.config.factor_v,
        )

    def solve_model(self, max_iters: int) -> None:
        convergence_result = np.inf
        master_δ = self.find_initial_state_of_switches()
        master_ζ = self.find_initial_state_of_taps()

        pbar = tqdm.tqdm(
            range(max_iters),
            desc="Master obj: 0, Slave obj: 0 and Gap: 1e6",
            disable=False,
        )
        logging.getLogger("pyomo.core").setLevel(logging.ERROR)
        for k in pbar:
            self.optimal_slave_model_instance.master_δ.store_values(master_δ)  # type: ignore
            self.scaled_optimal_slave_model_instance.master_δ.store_values(master_δ)  # type: ignore
            self.optimal_slave_model_instance.master_ζ.store_values(master_ζ)  # type: ignore
            self.scaled_optimal_slave_model_instance.master_ζ.store_values(master_ζ)  # type: ignore
            try:
                results = self.slave_solver.solve(
                    self.scaled_optimal_slave_model_instance, tee=self.config.verbose
                )
                quadratic_infeasible = False
            except gp.GurobiError as e:
                quadratic_infeasible = True

            if (
                results.solver.termination_condition == pyo.TerminationCondition.optimal
            ) and (not quadratic_infeasible):
                pyo.TransformationFactory("core.scale_model").propagate_solution(  # type: ignore
                    self.scaled_optimal_slave_model_instance,
                    self.optimal_slave_model_instance,
                )
                self.slave_obj = self.optimal_slave_model_instance.objective()  # type: ignore
                self.infeasible_slave = False
            else:
                self.infeasible_slave_model_instance.master_δ.store_values(master_δ)  # type: ignore
                self.scaled_infeasible_slave_model_instance.master_δ.store_values(master_δ)  # type: ignore
                self.infeasible_slave_model_instance.master_ζ.store_values(master_ζ)  # type: ignore
                self.scaled_infeasible_slave_model_instance.master_ζ.store_values(master_ζ)  # type: ignore
                _ = self.slave_solver.solve(
                    self.scaled_infeasible_slave_model_instance,
                    tee=self.config.verbose,
                )
                pyo.TransformationFactory("core.scale_model").propagate_solution(  # type: ignore
                    self.scaled_infeasible_slave_model_instance,
                    self.infeasible_slave_model_instance,
                )
                self.slave_obj = self.scaled_infeasible_slave_model_instance.scaled_objective()  # type: ignore
                self.check_slave_feasibility()
                self.infeasible_slave = True

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
            if convergence_result < self.config.convergence_threshold:
                break
            master_δ = self.master_model_instance.δ.extract_values()  # type: ignore
            master_ζ = self.master_model_instance.ζ.extract_values()  # type: ignore

    def add_benders_cut(self) -> None:
        constraint_name = ["master_switch_status_propagation"]
        if self.infeasible_slave == True:

            marginal_cost_df_original = pl.DataFrame(
                {
                    "name": list(dict(self.infeasible_slave_model_instance.dual).keys()),  # type: ignore
                    "marginal_cost": list(dict(self.infeasible_slave_model_instance.dual).values()),  # type: ignore
                }
            )
            δ_value = extract_optimization_results(
                self.infeasible_slave_model_instance, "master_δ"
            )
            ζ_value = extract_optimization_results(
                self.infeasible_slave_model_instance, "master_ζ"
            )

        else:
            marginal_cost_df_original = pl.DataFrame(
                {
                    "name": list(dict(self.optimal_slave_model_instance.dual).keys()),  # type: ignore
                    "marginal_cost": list(dict(self.optimal_slave_model_instance.dual).values()),  # type: ignore
                }
            )

            δ_value = extract_optimization_results(
                self.optimal_slave_model_instance, "master_δ"
            )
            ζ_value = extract_optimization_results(
                self.optimal_slave_model_instance, "master_ζ"
            )
        # Cut of switches
        marginal_cost_df = (
            marginal_cost_df_original.with_columns(
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

        marginal_cost_df = marginal_cost_df.join(δ_value, on="S").join(
            self.δ_variable, on="S"
        )

        self.marginal_cost = marginal_cost_df

        marginal_cost_df = marginal_cost_df.filter(c("master_δ") == 1)

        new_cut = self.slave_obj
        for data in marginal_cost_df.to_dicts():
            new_cut += data["marginal_cost"] * (data["δ_variable"] - data["master_δ"])

        if self.infeasible_slave == True:
            self.master_model_instance.infeasibility_cut.add(0 >= new_cut)  # type: ignore
        else:
            self.master_model_instance.optimality_cut.add(self.master_model_instance.θ1 >= new_cut)  # type: ignore
        # Cut of transformers
        constraint_name = ["master_transformer_status_propagation"]
        marginal_cost_df = (
            marginal_cost_df_original.with_columns(
                c("name").map_elements(lambda x: x.name, return_dtype=pl.Utf8),
            )
            .with_columns(
                c("name")
                .pipe(modify_string_col, format_str={"]": ""})
                .str.split("[")
                .list.to_struct(fields=["name", "TrTap"])
            )
            .unnest("name")
        )
        marginal_cost_df = (
            marginal_cost_df.filter(c("name").is_in(constraint_name))
            .select(
                c("name"),
                c("TrTap").str.split(",").list.to_struct(fields=["Tr", "tap"]),
                c("marginal_cost"),
            )
            .unnest("TrTap")
            .with_columns(c("Tr").cast(pl.Int64))
        ).with_columns(
            pl.concat_list(c("Tr").cast(pl.Int64), c("tap").cast(pl.Int64)).alias(
                "TrTaps"
            )
        )

        marginal_cost_df = marginal_cost_df.join(ζ_value, on="TrTaps").join(
            self.ζ_variable, on="TrTaps"
        )

        self.marginal_cost = marginal_cost_df

        new_cut = self.slave_obj
        for data in marginal_cost_df.to_dicts():
            new_cut += data["marginal_cost"] * (data["ζ_variable"] - data["master_ζ"])

        if self.infeasible_slave == True:
            self.master_model_instance.infeasibility_cut.add(0 >= new_cut)  # type: ignore
        else:
            self.master_model_instance.optimality_cut.add(self.master_model_instance.θ2 >= new_cut)  # type: ignore

    def check_slave_feasibility(self):
        self.p_curt_cons = extract_optimization_results(
            self.infeasible_slave_model_instance, "p_curt_cons"
        ).filter(c("p_curt_cons") > self.config.slack_threshold)

        self.q_curt_cons = extract_optimization_results(
            self.infeasible_slave_model_instance, "q_curt_cons"
        ).filter(c("q_curt_cons") > self.config.slack_threshold)

        self.p_curt_prod = extract_optimization_results(
            self.infeasible_slave_model_instance, "p_curt_prod"
        ).filter(c("p_curt_prod") > self.config.slack_threshold)
        self.q_curt_prod = extract_optimization_results(
            self.infeasible_slave_model_instance, "q_curt_prod"
        ).filter(c("q_curt_prod") > self.config.slack_threshold)

    def find_initial_state_of_switches(self):
        """
        Check if the topology is a tree.
        """
        nx_graph = nx.Graph()
        _ = self.data_manager.edge_data.filter(~c("normal_open")).with_columns(
            pl.struct(pl.all()).pipe(generate_nx_edge, nx_graph=nx_graph)
        )
        if nx.is_tree(nx_graph):

            slack_node_id = self.data_manager.node_data.filter(c("type") == "slack")[
                "node_id"
            ][0]
            digraph = generate_bfs_tree_with_edge_data(
                graph=nx_graph, source=slack_node_id
            )
            initial_master_δ = pl_to_dict_with_tuple(
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
            initial_master_δ = self.master_model_instance.δ.extract_values()  # type: ignore
        return initial_master_δ

    def find_initial_state_of_taps(self):
        results = self.master_solver.solve(self.master_model_instance, tee=False)
        if results.solver.termination_condition != pyo.TerminationCondition.optimal:
            log.warning(
                f"\nMaster model did not converge: {results.solver.termination_condition}"
            )

        self.master_obj = self.master_model_instance.objective()  # type: ignore
        initial_master_ζ = self.master_model_instance.ζ.extract_values()  # type: ignore
        return initial_master_ζ
