from typing import Dict, Tuple
import random
import pyomo.environ as pyo
import numpy as np
import polars as pl
from general_function import generate_log
from pipelines.data_manager import PipelineDataManager
from pipelines.configs import ADMMConfig, PipelineType
from optimization_model import generate_combined_model, generate_combined_lin_model
from pipelines.model_managers import PipelineModelManager
from tqdm import tqdm

log = generate_log(name=__name__)


class PipelineModelManagerADMM(PipelineModelManager):
    def __init__(self, config: ADMMConfig, data_manager: PipelineDataManager) -> None:
        """Initialize the combined model manager with configuration and data manager."""
        super().__init__(config, data_manager, PipelineType.ADMM)

        self.admm_model: pyo.AbstractModel = generate_combined_model()
        self.admm_linear_model: pyo.AbstractModel = generate_combined_lin_model()
        self.admm_linear_model_instance: pyo.ConcreteModel
        self.admm_model_instances: Dict[Tuple[int, int], pyo.ConcreteModel] = {}

        self.admm_obj: float = 0.0

        # ADMM artifacts
        self.admm_z: Dict[int, float] = {}
        self.admm_λ: Dict[tuple, float] = {}

    def instantaniate_model(self, grid_data_parameters_dict: dict | None) -> None:
        """Instantiate the ADMM model with the provided grid data parameters."""
        if grid_data_parameters_dict is None:
            raise ValueError("Grid data parameters dictionary cannot be None.")
        self.admm_linear_model_instance = self.admm_linear_model.create_instance(grid_data_parameters_dict[list(grid_data_parameters_dict.keys())[0]])  # type: ignore
        self.number_of_groups = len(
            grid_data_parameters_dict[list(grid_data_parameters_dict.keys())[0]][
                None
            ].get("groups", [])
        )
        for scen in grid_data_parameters_dict.keys():
            model = self.admm_model.create_instance(grid_data_parameters_dict[scen])
            for group in range(self.number_of_groups):
                self.admm_model_instances[(scen, group)] = model  # type: ignore

    def solve_model(
        self,
        max_iters: int = 50,
        ε_primal: float = 1e-3,
        ε_dual: float = 1e-3,
        μ: float = 10.0,
        τ_incr: float = 2.0,
        τ_decr: float = 2.0,
        seed_number: int = 42,
        κ: float = 0.1,
    ) -> None:
        """Solve the ADMM model with the given parameters."""

        ρ = self.config.ρ

        random.seed(seed_number)

        scen_list = list(self.admm_model_instances.keys())  # type: ignore

        # Sets
        switch_list = list(self.admm_model_instances[scen_list[0]].S)  # type: ignore
        scenario_number = len(scen_list)

        # Initialize consensus and duals
        z = {s: 0.5 for s in switch_list}  # consensus per switch
        λ = {(ω, s): 0.0 for ω in scen_list for s in switch_list}  # scaled duals
        results = self.solver.solve(
            self.admm_linear_model_instance, tee=self.config.verbose
        )
        δ_map = self.admm_linear_model_instance.δ.extract_values()  # type: ignore

        # ADMM iterations
        for k in range(1, max_iters + 1):
            # Broadcast z and λ into the model
            self._set_z_from_z(z)
            self._set_λ_from_λ(λ)
            # ---- x-update: solve each scenario with current z, λ ----
            δ_by_sc: Dict[Tuple[int, int], Dict[str, float]] = {}

            for ω in tqdm(scen_list, desc=f"ADMM iteration {k}/{max_iters}"):
                m = self.admm_model_instances[ω]  # type: ignore
                # Solve multi‑scenario instance
                random_number = random.randint(0, self.number_of_groups)

                for edge_id, δ in δ_map.items():
                    if random_number == 0:
                        m.δ[edge_id].unfix()  # type: ignore
                    else:
                        m.δ[edge_id].fix(1.0 if δ > 0.5 else 0.0)  # type: ignore

                δ_by_sc[ω] = δ_map
                try:
                    results = self.solver.solve(m, tee=self.config.verbose)
                    if (
                        results.solver.termination_condition
                        != pyo.TerminationCondition.optimal
                    ):
                        raise ValueError(
                            f"[ADMM {k}] solve failed: {results.solver.termination_condition}"
                        )
                    δ_map = m.δ.extract_values()  # type: ignore
                    δ_by_sc[ω] = δ_map
                except Exception as e:
                    log.error(f"[ADMM {k}] Error solving scenario {ω}: {e}")
                    continue

            # ---- z-update (per switch) ----
            z_old = z.copy()
            for s in switch_list:
                z[s] = (sum(δ_by_sc[ω][s] for ω in scen_list)) / scenario_number

            # ---- λ-update (scaled duals) ----
            for ω in scen_list:
                for s in switch_list:
                    λ[(ω, s)] = λ[(ω, s)] + κ * (δ_by_sc[ω][s] - z[s])

            # ---- residuals (after all scenarios solved) ----
            r_sq = max(
                [(δ_by_sc[ω][s] - z[s]) ** 2 for ω in scen_list for s in switch_list]
            )
            r_norm = float(np.sqrt(r_sq))

            s_sq = sum((z[s] - z_old[s]) ** 2 for s in switch_list)
            s_norm = float(self.config.ρ * np.sqrt(s_sq))

            log.info(
                f"[ADMM {k}] r={r_norm:.3e}, s={s_norm:.3e}, ρ={self.config.ρ:.3g}"
            )

            # ---- convergence ----
            if (r_norm <= ε_primal) and (s_norm <= ε_dual):
                log.info(f"ADMM converged in {k} iterations.")
                break

            # ---- residual balancing (scaled ADMM) ----
            if r_norm > μ * s_norm:
                self.config.ρ *= τ_incr
            elif s_norm > μ * r_norm:
                self.config.ρ /= τ_decr
            for m in self.admm_model_instances.values():
                getattr(m, "ρ").set_value(self.config.ρ)

        # Store results
        self.admm_z = z
        self.admm_λ = λ

        # Refresh δ table after final iterate
        rows = []
        for ω in scen_list:
            m = self.admm_model_instances[ω]
            δ_map = m.δ.extract_values()  # type: ignore
            for s in switch_list:
                rows.append((ω, s, float(δ_map[s])))
        self.δ_variable = pl.DataFrame(
            rows,
            schema=["SCEN", "S", "δ_variable"],
            orient="row",
        )

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

    def _set_z_from_z(self, z_per_switch: dict) -> None:
        """Broadcast consensus z[s] to del_param[sc, s] for all scenarios sc."""
        for _, m in self.admm_model_instances.items():
            z_param = getattr(m, "z")
            for s in getattr(m, "S"):
                z_param[s].set_value(z_per_switch[s])

    def _set_λ_from_λ(self, λ_map: dict) -> None:
        """Set per-scenario scaled duals λ_sc[s] from λ_map[(scen_id, s)]."""
        for ω, m in self.admm_model_instances.items():
            λ_param = getattr(m, "λ")
            for s in getattr(m, "S"):
                λ_param[s].set_value(λ_map[(ω, s)])
