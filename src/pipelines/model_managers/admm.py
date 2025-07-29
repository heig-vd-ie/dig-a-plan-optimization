from typing import Dict
import pyomo.environ as pyo
import numpy as np
import polars as pl
from general_function import generate_log
from pipelines.data_manager import PipelineDataManager
from pipelines.configs import ADMMConfig, PipelineType
from optimization_model import generate_combined_model
from pipelines.model_managers import PipelineModelManager

log = generate_log(name=__name__)


class PipelineModelManagerADMM(PipelineModelManager):
    def __init__(self, config: ADMMConfig, data_manager: PipelineDataManager) -> None:
        """Initialize the combined model manager with configuration and data manager."""
        super().__init__(config, data_manager, PipelineType.ADMM)

        self.admm_model: pyo.AbstractModel = generate_combined_model()
        self.admm_model_instances: Dict[str, pyo.ConcreteModel] = {}
        self.scaled_admm_model_instance: Dict[str, pyo.ConcreteModel] = {}

        self.admm_obj: float = 0.0

        # ADMM artifacts
        self.admm_z: Dict[str, float] = {}  # consensus per switch: {s: z_s}
        self.admm_u: Dict[tuple, float] = {}  # scaled duals: {(sc, s): u_sc,s}

    def instantaniate_model(self, grid_data_parameters_dict: dict | None) -> None:
        """Instantiate the ADMM model with the provided grid data parameters."""
        if grid_data_parameters_dict is None:
            raise ValueError("Grid data parameters dictionary cannot be None.")
        for scen in grid_data_parameters_dict.keys():
            self.admm_model_instances[scen] = self.admm_model.create_instance(grid_data_parameters_dict[scen])  # type: ignore

    def solve_model(
        self,
        ρ: float = 1.0,
        max_iters: int = 50,
        eps_primal: float = 1e-3,
        eps_dual: float = 1e-3,
        adapt_ρ: bool = True,
        mu: float = 10.0,
        tau_incr: float = 2.0,
        tau_decr: float = 2.0,
    ) -> None:
        """
        Run consensus‑ADMM on δ[SCEN, S]. Assumes the model objective is ADMM_objective_rule
        and the model defines: ρ (Param), del_param[SCEN,S], u_param[SCEN,S].
        """
        models = self.admm_model_instances
        scen_list = list(models.keys())  # type: ignore

        # Sets
        switch_list = list(models[scen_list[0]].S)  # type: ignore
        scenario_number = len(scen_list)

        # Initialize consensus and duals
        z = {s: 0.5 for s in switch_list}  # consensus per switch
        λ = {(sc, s): 0.0 for sc in scen_list for s in switch_list}  # scaled duals

        # ADMM iterations
        for k in range(1, max_iters + 1):
            # Broadcast z and λ into the model
            self._set_z_from_z(z)
            self._set_λ_from_λ(λ)
            for scen, m in models.items():
                # Solve multi‑scenario instance
                results = self.solver.solve(m, tee=self.config.verbose)
                if (
                    results.solver.termination_condition
                    != pyo.TerminationCondition.optimal
                ):
                    raise ValueError(
                        f"[ADMM {k}] solve failed: {results.solver.termination_condition}"
                    )

                # Gather local δ
                scen, sw, δ_mat = self._extract_δ_matrix()

                # z‑update (per switch)
                z_old = z.copy()
                for j, s in enumerate(sw):
                    z[s] = (
                        δ_mat[:, j].sum() + sum(z[(sc, s)] for sc in scen)
                    ) / scenario_number

                # u‑update
                for i, sc in enumerate(scen):
                    for j, s in enumerate(sw):
                        z[(sc, s)] = z[(sc, s)] + δ_mat[i, j] - z[s]

                # Residuals
                r_sq = 0.0
                for i, sc in enumerate(scen):
                    for j, s in enumerate(sw):
                        r_sq += (δ_mat[i, j] - z[s]) ** 2
                r_norm = np.sqrt(r_sq)

                s_sq = sum((z[s] - z_old[s]) ** 2 for s in sw)
                s_norm = ρ * np.sqrt(scenario_number * s_sq)

                log.info(f"[ADMM {k}] r={r_norm:.3e}, s={s_norm:.3e}, ρ={ρ:.3g}")

                # Convergence
                if (r_norm <= eps_primal) and (s_norm <= eps_dual):
                    log.info(f"ADMM converged in {k} iterations.")
                    break

                # Residual balancing (scaled ADMM)
                if adapt_ρ:
                    if r_norm > mu * s_norm:
                        ρ *= tau_incr
                        m.ρ.set_value(ρ)  # type: ignore
                    elif s_norm > mu * r_norm:
                        ρ /= tau_decr
                        m.ρ.set_value(ρ)  # type: ignore

            # Store results
            self.admm_z = z
            self.admm_λ = λ

            # Refresh δ table after final iterate
            scen, sw, δ_mat = self._extract_δ_matrix()
            self.δ_variable = pl.DataFrame(
                {
                    "SCEN": [sc for sc in scen for _ in sw],
                    "S": [s for _ in scen for s in sw],
                    "δ_variable": [
                        δ_mat[i, j]
                        for i, _ in enumerate(scen)
                        for j, _ in enumerate(sw)
                    ],
                }
            )
            # store objective value
            final_obj = pyo.value(self.combined_model_instance.objective)  # type: ignore
            self.combined_obj_list.append(final_obj)  # type: ignore
            log.info(f"ADMM final objective = {final_obj:.4f}")

    # ---------------------------
    # ADMM helpers and main loop
    # ---------------------------

    def _extract_δ_matrix(self) -> tuple[list, list, np.ndarray]:
        """Return (scen_list, switch_list, δ[sc, s] as 2D np.array)."""
        m = self.admm_model_instances
        scen = list(m.SCEN)  # type: ignore
        sw = list(m.S)  # type: ignore
        arr = np.zeros((len(scen), len(sw)))
        for i, sc in enumerate(scen):
            for j, s in enumerate(sw):
                arr[i, j] = pyo.value(m.δ[sc, s])  # type: ignore
        return scen, sw, arr

    def _set_z_from_z(self, z_per_switch: dict) -> None:
        """Broadcast consensus z[s] to del_param[sc, s] for all scenarios sc."""
        m = self.admm_model_instances
        for sc in m.SCEN:  # type: ignore
            for s in m.S:  # type: ignore
                m.z[sc, s].set_value(z_per_switch[s])  # type: ignore

    def _set_λ_from_λ(self, λ_map: dict) -> None:
        """Set λ[sc, s] from dictionary λ_map[(sc, s)]."""
        m = self.admm_model_instances
        for sc in m.SCEN:  # type: ignore
            for s in m.S:  # type: ignore
                m.λ[sc, s].set_value(u_map[(sc, s)])  # type: ignore
