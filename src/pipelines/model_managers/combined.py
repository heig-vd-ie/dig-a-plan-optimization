import pyomo.environ as pyo
import polars as pl
from general_function import generate_log
from polars import col as c
from polars_function import cast_boolean
from pipelines.data_manager import PipelineDataManager
from pipelines.configs import CombinedConfig, PipelineType
from optimization_model import generate_combined_model
from pyomo_utility import extract_optimization_results

log = generate_log(name=__name__)


class PipelineModelManagerCombined:
    def __init__(
        self, config: CombinedConfig, data_manager: PipelineDataManager
    ) -> None:
        """Initialize the combined model manager with configuration and data manager."""
        if config.pipeline_type != PipelineType.COMBINED:
            raise ValueError(
                f"Pipeline type must be {PipelineType.COMBINED}, got {config.pipeline_type}"
            )

        self.config = config
        self.data_manager = data_manager

        self.delta_variable: pl.DataFrame
        self.combined_model: pyo.AbstractModel = generate_combined_model()
        self.combined_model_instance: pyo.ConcreteModel
        self.scaled_combined_model_instance: pyo.ConcreteModel

        self.d: pl.DataFrame = pl.DataFrame()
        self.combined_obj: float = 0.0

        self.combined_solver = pyo.SolverFactory(config.combined_solver_name)
        self.combined_solver.options["IntegralityFocus"] = (
            config.combined_solver_integrality_focus
        )  # To insure master binary variable remains binary
        if config.combined_solver_non_convex is not None:
            self.combined_solver.options["NonConvex"] = (
                config.combined_solver_non_convex
            )
        if config.combined_solver_qcp_dual is not None:
            self.combined_solver.options["QCPDual"] = config.combined_solver_qcp_dual
        if config.combined_solver_bar_qcp_conv_tol is not None:
            self.combined_solver.options["BarQCPConvTol"] = (
                config.combined_solver_bar_qcp_conv_tol
            )
        if config.combined_solver_bar_homogeneous is not None:
            self.combined_solver.options["BarHomogeneous"] = (
                config.combined_solver_bar_homogeneous
            )

        # Build the combined model
        self.combined_model: pyo.AbstractModel = generate_combined_model()
        self.combined_model_instance: pyo.ConcreteModel

        # Solver

        self.combined_obj_list: list[float] = []

    def instantaniate_model(self, grid_data_parameters_dict: dict | None) -> None:
        self.combined_model_instance = self.combined_model.create_instance(grid_data_parameters_dict)  # type: ignore
        self.delta_variable = pl.DataFrame(
            self.combined_model_instance.delta.items(),  # type: ignore
            schema=["S", "delta_variable"],
        )

    def solve_model(self, **kwargs) -> None:
        """Solve the combined radial+DistFlow model."""
        results = self.combined_solver.solve(
            self.combined_model_instance, tee=self.config.verbose
        )
        if results.solver.termination_condition != pyo.TerminationCondition.optimal:
            log.error(f"Solve failed: {results.solver.termination_condition}")
            return
        current_obj = pyo.value(self.combined_model_instance.objective)
        self.combined_obj_list.append(current_obj)  # type: ignore
        log.info(f"Combined solve successful: objective = {current_obj:.4f}")
        self.delta_variable = pl.DataFrame(
            self.combined_model_instance.delta.items(),  # type: ignore
            schema=["S", "delta_variable"],
        )
