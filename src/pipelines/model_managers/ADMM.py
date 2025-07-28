import pyomo.environ as pyo
import numpy as np
from typing import Any, cast
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
        self.combined_solver.options["Method"] = 2
        self.combined_solver.options["TimeLimit"] = 60
        self.combined_solver.options["OptimalityTol"] = 1e-5
        self.combined_solver.options["FeasibilityTol"] = 1e-5
        self.combined_solver.options["BarConvTol"] = 1e-5
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

        # ADMM artifacts
        self.admm_z: dict | None = None  # consensus per switch: {s: z_s}
        self.admm_u: dict | None = None  # scaled duals: {(sc, s): u_sc,s}

    def instantaniate_model(self, grid_data_parameters_dict: dict | None) -> None:
        self.combined_model_instance = self.combined_model.create_instance(grid_data_parameters_dict)  # type: ignore
        # delta is indexed by (SCEN, S); extract both indices
        m = self.combined_model_instance
        scen = list(m.SCEN)  # type: ignore
        sw = list(m.S)  # type: ignore
        records = []
        for sc in scen:
            for s in sw:
                records.append((sc, s, pyo.value(m.delta[sc, s])))  # type: ignore
        self.delta_variable = pl.DataFrame(
            records, schema=["SCEN", "S", "delta_variable"]
        )

    # ---------------------------
    # ADMM helpers and main loop
    # ---------------------------

    def _extract_delta_matrix(self) -> tuple[list, list, np.ndarray]:
        """Return (scen_list, switch_list, delta[sc, s] as 2D np.array)."""
        m = self.combined_model_instance
        scen = list(m.SCEN)  # type: ignore
        sw = list(m.S)  # type: ignore
        arr = np.zeros((len(scen), len(sw)))
        for i, sc in enumerate(scen):
            for j, s in enumerate(sw):
                arr[i, j] = pyo.value(m.delta[sc, s])  # type: ignore
        return scen, sw, arr

    def _set_del_param_from_z(self, z_per_switch: dict) -> None:
        """Broadcast consensus z[s] to del_param[sc, s] for all scenarios sc."""
        m = self.combined_model_instance
        for sc in m.SCEN:  # type: ignore
            for s in m.S:  # type: ignore
                m.del_param[sc, s].set_value(z_per_switch[s])  # type: ignore

    def _set_u_param(self, u_map: dict) -> None:
        """Set u_param[sc, s] from dictionary u_map[(sc, s)]."""
        m = self.combined_model_instance
        for sc in m.SCEN:  # type: ignore
            for s in m.S:  # type: ignore
                m.u_param[sc, s].set_value(u_map[(sc, s)])  # type: ignore

    def solve_with_admm(
        self,
        rho: float = 1.0,
        max_iters: int = 50,
        eps_primal: float = 1e-3,
        eps_dual: float = 1e-3,
        adapt_rho: bool = True,
        mu: float = 10.0,
        tau_incr: float = 2.0,
        tau_decr: float = 2.0,
    ) -> None:
        """
        Run consensus‑ADMM on delta[SCEN, S]. Assumes the model objective is ADMM_objective_rule
        and the model defines: rho (Param), del_param[SCEN,S], u_param[SCEN,S].
        """
        m = self.combined_model_instance

        # Sets
        scen_list = list(m.SCEN)  # type: ignore
        sw_list = list(m.S)  # type: ignore
        S = len(scen_list)

        # Initialize consensus and duals
        z = {s: 0.5 for s in sw_list}  # consensus per switch
        u = {(sc, s): 0.0 for sc in scen_list for s in sw_list}  # scaled duals

        # Set rho in the model (single scalar for all terms)
        if hasattr(m, "rho"):
            m.rho.set_value(rho)  # type: ignore

        # ADMM iterations
        for k in range(1, max_iters + 1):
            # Broadcast z and u into the model
            self._set_del_param_from_z(z)
            self._set_u_param(u)

            # Solve multi‑scenario instance
            results = self.combined_solver.solve(m, tee=self.config.verbose)
            if results.solver.termination_condition != pyo.TerminationCondition.optimal:
                log.error(
                    f"[ADMM {k}] solve failed: {results.solver.termination_condition}"
                )
                break

            # Gather local delta
            scen, sw, delta_mat = self._extract_delta_matrix()

            # z‑update (per switch)
            z_old = z.copy()
            for j, s in enumerate(sw):
                z[s] = (delta_mat[:, j].sum() + sum(u[(sc, s)] for sc in scen)) / S

            # u‑update
            for i, sc in enumerate(scen):
                for j, s in enumerate(sw):
                    u[(sc, s)] = u[(sc, s)] + delta_mat[i, j] - z[s]

            # Residuals
            r_sq = 0.0
            for i, sc in enumerate(scen):
                for j, s in enumerate(sw):
                    r_sq += (delta_mat[i, j] - z[s]) ** 2
            r_norm = np.sqrt(r_sq)

            s_sq = sum((z[s] - z_old[s]) ** 2 for s in sw)
            s_norm = rho * np.sqrt(S * s_sq)

            log.info(f"[ADMM {k}] r={r_norm:.3e}, s={s_norm:.3e}, rho={rho:.3g}")

            # Convergence
            if (r_norm <= eps_primal) and (s_norm <= eps_dual):
                log.info(f"ADMM converged in {k} iterations.")
                break

            # Residual balancing (scaled ADMM)
            if adapt_rho:
                if r_norm > mu * s_norm:
                    rho *= tau_incr
                    m.rho.set_value(rho)  # type: ignore
                elif s_norm > mu * r_norm:
                    rho /= tau_decr
                    m.rho.set_value(rho)  # type: ignore

        # Store results
        self.admm_z = z
        self.admm_u = u

        # Refresh delta table after final iterate
        scen, sw, delta_mat = self._extract_delta_matrix()
        self.delta_variable = pl.DataFrame(
            {
                "SCEN": [sc for sc in scen for _ in sw],
                "S": [s for _ in scen for s in sw],
                "delta_variable": [
                    delta_mat[i, j]
                    for i, _ in enumerate(scen)
                    for j, _ in enumerate(sw)
                ],
            }
        )
        # store objective value
        final_obj = pyo.value(self.combined_model_instance.objective)  # type: ignore
        self.combined_obj_list.append(final_obj)  # type: ignore
        log.info(f"ADMM final objective = {final_obj:.4f}")
