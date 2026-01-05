import pyomo.environ as pyo
import polars as pl
from helpers import generate_log
from pipeline_reconfiguration.data_manager import PipelineDataManager
from data_model.reconfiguration_konfig import PipelineConfig

log = generate_log(name=__name__)


class PipelineModelManager:
    def __init__(
        self,
        konfig: PipelineConfig,
        data_manager: PipelineDataManager,
    ) -> None:

        self.konfig = konfig
        self.data_manager = data_manager

        self.δ_variable: pl.DataFrame

        self.solver = pyo.SolverFactory(konfig.solver_name)
        self.solver.options["IntegralityFocus"] = (
            konfig.solver_integrality_focus
        )  # To insure master binary variable remains binary
        self.solver.options["Seed"] = konfig.seed
        if konfig.threads is not None:
            self.solver.options["Threads"] = konfig.threads
        self.solver.options["Method"] = konfig.solver_method
        self.solver.options["TimeLimit"] = konfig.time_limit
        self.solver.options["OptimalityTol"] = konfig.optimality_tolerance
        self.solver.options["FeasibilityTol"] = konfig.feasibility_tolerance
        self.solver.options["BarConvTol"] = konfig.barrier_convergence_tolerance
        if konfig.solver_non_convex is not None:
            self.solver.options["NonConvex"] = konfig.solver_non_convex
        if konfig.solver_qcp_dual is not None:
            self.solver.options["QCPDual"] = konfig.solver_qcp_dual
        if konfig.solver_bar_qcp_conv_tol is not None:
            self.solver.options["BarQCPConvTol"] = konfig.solver_bar_qcp_conv_tol
        if konfig.solver_bar_homogeneous is not None:
            self.solver.options["BarHomogeneous"] = konfig.solver_bar_homogeneous
