from dataclasses import dataclass
from enum import Enum


class PipelineType(Enum):
    """Enum for different pipeline types"""

    BENDER = "bender"
    COMBINED = "combined"
    ADMM = "admm"


@dataclass
class PipelineConfig:
    """Configuration for the Dig A Plan optimization pipeline"""

    verbose: bool = False
    voll: float = 1.0
    volp: float = 1.0
    ρ: float = 10.0
    big_m: float = 1e4
    ε: float = 1
    slack_threshold: float = 1e-2
    convergence_threshold: float = 1e-4
    pipeline_type: PipelineType = PipelineType.BENDER
    max_iterations: int = 100
    factor_p: float = 1.0
    factor_q: float = 1.0
    factor_i: float = 1.0
    factor_v: float = 1.0
    γ_infeasibility: float = 1.0
    γ_admm_penalty: float = 1.0
    γ_trafo_loss: float = 1.0
    solver_name: str = "gurobi"
    solver_integrality_focus: int = 1
    solver_method: int = 2
    time_limit: int = 60
    optimality_tolerance: float = 1e-5
    feasibility_tolerance: float = 1e-5
    barrier_convergence_tolerance: float = 1e-5
    solver_non_convex: int | None = None
    solver_qcp_dual: int | None = None
    solver_bar_qcp_conv_tol: float | None = None
    solver_bar_homogeneous: int | None = None
    all_scenarios: bool = False
    seed: int = 1234
    threads: int | None = None


@dataclass
class BenderConfig(PipelineConfig):
    """Configuration for Bender's decomposition pipeline"""

    # Solver configurations
    master_relaxed: bool = False
    master_solver_options: dict | None = None
    slave_solver_options: dict | None = None
    master_solver_integrality_focus: int = 1


@dataclass
class CombinedConfig(PipelineConfig):
    """Configuration for Bender's decomposition pipeline"""

    # Solver configurations
    pass


@dataclass
class ADMMConfig(PipelineConfig):
    """Configuration for ADMM pipeline"""

    admm_max_iterations: int = 10
    admm_tolerance: float = 1e-4
    max_iters: int = 10
    μ: float = 10.0
    τ_incr: float = 2.0
    τ_decr: float = 2.0
    mutation_factor: int = 5
    groups: int | dict[int, list[int]] = 10
    ε_primal: float = 1e-3
    ε_dual: float = 1e-3
    seed_number: int = 42
    κ: float = 0.1
