from dataclasses import dataclass
from enum import Enum


class PipelineType(Enum):
    """Enum for different pipeline types"""

    BENDER = "bender"
    COMBINED = "combined"


@dataclass
class PipelineConfig:
    """Configuration for the Dig A Plan optimization pipeline"""

    verbose: bool = False
    big_m: float = 1e4
    slack_threshold: float = 1e-5
    convergence_threshold: float = 1e-4
    pipeline_type: PipelineType = PipelineType.BENDER
    max_iterations: int = 100
    factor_p: float = 1.0
    factor_q: float = 1.0
    factor_i: float = 1.0
    factor_v: float = 1.0


@dataclass
class BenderConfig(PipelineConfig):
    """Configuration for Bender's decomposition pipeline"""

    # Solver configurations
    master_relaxed: bool = False
    master_solver_options: dict | None = None
    slave_solver_options: dict | None = None
    master_solver_name: str = "gurobi"
    slave_solver_name: str = "gurobi"
    master_solver_integrality_focus: int = 1
    slave_solver_non_convex: int | None = None
    slave_solver_qcp_dual: int | None = None
    slave_solver_bar_qcp_conv_tol: float | None = None
    slave_solver_bar_homogeneous: int | None = None


@dataclass
class CombinedConfig(PipelineConfig):
    """Configuration for Bender's decomposition pipeline"""

    # Solver configurations
    combined_solver_options: dict | None = None
    combined_solver_name: str = "gurobi"
    combined_solver_integrality_focus: int = 1
    combined_solver_non_convex: int | None = None
    combined_solver_qcp_dual: int | None = None
    combined_solver_bar_qcp_conv_tol: float | None = None
    combined_solver_bar_homogeneous: int | None = None
