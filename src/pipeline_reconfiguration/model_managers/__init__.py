import pyomo.environ as pyo
import polars as pl
from helpers import generate_log
from pipeline_reconfiguration.data_manager import PipelineDataManager
from pipeline_reconfiguration.configs import PipelineConfig

log = generate_log(name=__name__)


class PipelineModelManager:
    def __init__(
        self,
        config: PipelineConfig,
        data_manager: PipelineDataManager,
    ) -> None:

        self.config = config
        self.data_manager = data_manager

        self.δ_variable: pl.DataFrame

        self.solver = pyo.SolverFactory(config.solver_name)
        self.solver.options["IntegralityFocus"] = (
            config.solver_integrality_focus
        )  # To insure master binary variable remains binary
        self.solver.options["Seed"] = config.seed
        if config.threads is not None:
            self.solver.options["Threads"] = config.threads
        self.solver.options["Method"] = config.solver_method
        self.solver.options["TimeLimit"] = config.time_limit
        self.solver.options["OptimalityTol"] = config.optimality_tolerance
        self.solver.options["FeasibilityTol"] = config.feasibility_tolerance
        self.solver.options["BarConvTol"] = config.barrier_convergence_tolerance
        if config.solver_non_convex is not None:
            self.solver.options["NonConvex"] = config.solver_non_convex
        if config.solver_qcp_dual is not None:
            self.solver.options["QCPDual"] = config.solver_qcp_dual
        if config.solver_bar_qcp_conv_tol is not None:
            self.solver.options["BarQCPConvTol"] = config.solver_bar_qcp_conv_tol
        if config.solver_bar_homogeneous is not None:
            self.solver.options["BarHomogeneous"] = config.solver_bar_homogeneous
