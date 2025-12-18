from pydantic import BaseModel, Field


class PipelineConfig(BaseModel):
    """Configuration for the Dig A Plan optimization pipeline"""

    verbose: bool = Field(
        default=False,
        description="Enable verbose logging and detailed solver output during execution.",
    )
    voll: float = Field(
        default=1.0,
        description="Penalty coefficient for loss of load in the objective function.",
    )
    volp: float = Field(
        default=1.0,
        description="Penalty coefficient for loss of production or curtailed generation.",
    )
    big_m: float = Field(
        default=1e4,
        description="Big-M constant used in linearization of logical or conditional constraints.",
    )
    ε: float = Field(
        default=1.0,
        description="Small numerical constant used to relax strict inequalities or avoid degeneracy.",
    )
    ρ: float = Field(
        default=10.0,
        description="Augmented Lagrangian penalty parameter used in decomposition methods.",
    )
    slack_threshold: float = Field(
        default=1e-2,
        description="Maximum allowed slack before a constraint is considered violated.",
    )
    convergence_threshold: float = Field(
        default=1e-4,
        description="Stopping threshold for convergence based on objective or residual change.",
    )
    factor_q: float = Field(
        default=1.0,
        description="Scaling factor applied to reactive power related terms.",
    )
    factor_i: float = Field(
        default=1.0, description="Scaling factor applied to current related terms."
    )
    factor_v: float = Field(
        default=1.0, description="Scaling factor applied to voltage related terms."
    )
    γ_infeasibility: float = Field(
        default=1.0,
        description="Weight applied to infeasibility penalties in the objective.",
    )
    γ_admm_penalty: float = Field(
        default=1.0,
        description="Weight applied to ADMM penalty terms in the objective.",
    )
    γ_trafo_loss: float = Field(
        default=1.0,
        description="Weight applied to transformer loss terms in the objective.",
    )
    solver_name: str = Field(
        default="gurobi", description="Name of the optimization solver to be used."
    )
    solver_integrality_focus: int = Field(
        default=1,
        description="Solver setting controlling emphasis on integrality feasibility.",
    )
    solver_method: int = Field(
        default=2, description="Solver algorithm selection parameter, solver-specific."
    )
    time_limit: int = Field(
        default=60,
        description="Maximum wall-clock time allowed for the solver in seconds.",
    )
    optimality_tolerance: float = Field(
        default=1e-5, description="Tolerance for declaring optimality of the solution."
    )
    feasibility_tolerance: float = Field(
        default=1e-5, description="Tolerance for constraint feasibility checks."
    )
    barrier_convergence_tolerance: float = Field(
        default=1e-5,
        description="Convergence tolerance for barrier or interior-point methods.",
    )
    solver_non_convex: int | None = Field(
        default=None,
        description="Flag controlling solver handling of non-convex models.",
    )
    solver_qcp_dual: int | None = Field(
        default=None,
        description="Enable or disable dual information for quadratic constraint programs.",
    )
    solver_bar_qcp_conv_tol: float | None = Field(
        default=None,
        description="Barrier convergence tolerance specific to QCP problems.",
    )
    solver_bar_homogeneous: int | None = Field(
        default=None,
        description="Enable homogeneous self-dual formulation in barrier solver.",
    )
    all_scenarios: bool = Field(
        default=False,
        description="If true, solve the model over all scenarios simultaneously.",
    )
    threads: int | None = Field(
        default=None,
        description="Number of solver threads to use; defaults to solver choice if unset.",
    )


class BenderConfig(PipelineConfig):
    """Configuration for Bender's decomposition pipeline"""

    # Solver configurations
    master_relaxed: bool = Field(
        default=False,
        description="If true, relax integrality constraints in the master problem.",
    )
    master_solver_options: dict | None = Field(
        default=None,
        description="Additional solver-specific options for the master problem.",
    )
    slave_solver_options: dict | None = Field(
        default=None,
        description="Additional solver-specific options for the slave or subproblems.",
    )
    master_solver_integrality_focus: int = Field(
        default=1,
        description="Integrality focus parameter applied only to the master problem.",
    )
    max_iters: int = Field(
        default=100, description="Maximum number of Bender iterations."
    )


class CombinedConfig(PipelineConfig):
    """Configuration for Bender's decomposition pipeline"""

    # Solver configurations
    groups: int | dict[int, list[int]] | None = Field(
        default=None,
        description="Number of ADMM groups or explicit mapping of group indices to variables.",
    )


class ADMMConfig(PipelineConfig):
    """Configuration for ADMM pipeline"""

    max_iters: int = Field(default=10, description="Maximum number of ADMM iterations.")
    admm_tolerance: float = Field(
        default=1e-4,
        description="Stopping tolerance for ADMM primal and dual residuals.",
    )
    μ: float = Field(default=10.0, description="Initial penalty parameter for ADMM.")
    τ_incr: float = Field(
        default=2.0,
        description="Factor by which the ADMM penalty parameter is increased.",
    )
    τ_decr: float = Field(
        default=2.0,
        description="Factor by which the ADMM penalty parameter is decreased.",
    )
    mutation_factor: int = Field(
        default=5,
        description="Number of iterations between adaptive parameter updates.",
    )
    groups: int | dict[int, list[int]] = Field(
        default=10,
        description="Number of ADMM groups or explicit mapping of group indices to variables.",
    )
    ε_primal: float = Field(
        default=1e-3, description="Tolerance for ADMM primal residual convergence."
    )
    ε_dual: float = Field(
        default=1e-3, description="Tolerance for ADMM dual residual convergence."
    )
    κ: float = Field(
        default=0.1, description="Relaxation or damping parameter used in ADMM updates."
    )
