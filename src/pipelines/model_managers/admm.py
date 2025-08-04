from typing import Dict
import pyomo.environ as pyo
import numpy as np
import polars as pl
from general_function import generate_log
from pipelines.data_manager import PipelineDataManager
from pipelines.configs import ADMMConfig, PipelineType
from optimization_model import generate_combined_model
from pipelines.model_managers import PipelineModelManager
from tqdm import tqdm

log = generate_log(name=__name__)


class PipelineModelManagerADMM(PipelineModelManager):
    def __init__(self, config: ADMMConfig, data_manager: PipelineDataManager) -> None:
        """Initialize the combined model manager with configuration and data manager."""
        super().__init__(config, data_manager, PipelineType.ADMM)

        self.admm_model: pyo.AbstractModel = generate_combined_model()
        self.admm_model_instances: Dict[int, pyo.ConcreteModel] = {}
        self.scaled_admm_model_instance: Dict[int, pyo.ConcreteModel] = {}

        self.admm_obj: float = 0.0

        # ADMM artifacts
        self.admm_z: Dict[int, float] = {}  # consensus per switch: {s: z_s}
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
            # ---- x-update: solve each scenario with current z, λ ----
            δ_by_sc: dict[int, np.ndarray] = (
                {}
            )  # {scenario_id: np.array over switch_list}

            for sc in tqdm(scen_list, desc=f"ADMM iteration {k}/{max_iters}"):
                m = models[sc]  # type: ignore
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
                # Gather local δ (vector over S)
                sw, δ_vec = self._extract_δ_vector(m)
                δ_by_sc[sc] = δ_vec

            # ---- z-update (per switch) ----
            z_old = z.copy()
            for j, s in enumerate(switch_list):
                z[s] = (
                    sum(δ_by_sc[sc][j] + λ[(sc, s)] for sc in scen_list)
                ) / scenario_number

            # ---- λ-update (scaled duals) ----
            for sc in scen_list:
                for j, s in enumerate(switch_list):
                    λ[(sc, s)] = λ[(sc, s)] + δ_by_sc[sc][j] - z[s]

            # ---- residuals (after all scenarios solved) ----
            r_sq = sum(
                (δ_by_sc[sc][j] - z[s]) ** 2
                for sc in scen_list
                for j, s in enumerate(switch_list)
            )
            r_norm = float(np.sqrt(r_sq))

            s_sq = sum((z[s] - z_old[s]) ** 2 for s in switch_list)
            s_norm = float(ρ * np.sqrt(scenario_number * s_sq))

            log.info(f"[ADMM {k}] r={r_norm:.3e}, s={s_norm:.3e}, ρ={ρ:.3g}")

            # ---- convergence ----
            if (r_norm <= eps_primal) and (s_norm <= eps_dual):
                log.info(f"ADMM converged in {k} iterations.")
                break

            # ---- residual balancing (scaled ADMM) ----
            if adapt_ρ:
                if r_norm > mu * s_norm:
                    ρ *= tau_incr
                elif s_norm > mu * r_norm:
                    ρ /= tau_decr
                for m in models.values():
                    getattr(m, "ρ").set_value(ρ)

        # Store results
        self.admm_z = z
        self.admm_λ = λ

        # Refresh δ table after final iterate
        rows = []
        for sc in scen_list:
            m = models[sc]
            sw, δ_vec = self._extract_δ_vector(m)
            for j, s in enumerate(sw):
                rows.append((sc, s, float(δ_vec[j])))
        self.δ_variable = pl.DataFrame(rows, schema=["SCEN", "S", "δ_variable"])

        # store total objective (sum over scenarios), if available
        try:
            final_obj = float(sum(pyo.value(getattr(m, "objective")) for m in models.values()))  # type: ignore
            self.combined_obj_list.append(final_obj)  # type: ignore[attr-defined]
            log.info(f"ADMM final objective (sum over scenarios) = {final_obj:.4f}")
        except Exception:
            pass

    # ---------------------------
    # ADMM helpers and main loop
    # ---------------------------

    # def _extract_δ_matrix(self) -> tuple[list, list, np.ndarray]:
    #     """Return (scen_list, switch_list, δ[sc, s] as 2D np.array)."""
    #     m = self.admm_model_instances
    #     scen = list(m.SCEN)  # type: ignore
    #     sw = list(m.S)  # type: ignore
    #     arr = np.zeros((len(scen), len(sw)))
    #     for i, sc in enumerate(scen):
    #         for j, s in enumerate(sw):
    #             arr[i, j] = pyo.value(m.δ[sc, s])  # type: ignore
    #     return scen, sw, arr

    def _set_z_from_z(self, z_per_switch: dict) -> None:
        """Broadcast consensus z[s] to del_param[sc, s] for all scenarios sc."""
        for scen_id, m in self.admm_model_instances.items():
            z_param = getattr(m, "z")
            for s in getattr(m, "S"):
                z_param[s].set_value(z_per_switch[s])

    def _set_λ_from_λ(self, λ_map: dict) -> None:
        """Set per-scenario scaled duals λ_sc[s] from λ_map[(scen_id, s)]."""
        for scen_id, m in self.admm_model_instances.items():
            lam_param = getattr(m, "λ")
            for s in getattr(m, "S"):
                lam_param[s].set_value(λ_map[(scen_id, s)])

    def _extract_δ_vector(self, m) -> tuple[list, np.ndarray]:
        """Return (switch_list, δ[s] as 1D np.array) for a single scenario model."""
        δ = getattr(m, "δ")
        sw = list(getattr(m, "S"))
        vec = np.array([pyo.value(δ[s]) for s in sw], dtype=float)
        return sw, vec
