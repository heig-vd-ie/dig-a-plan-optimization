import itertools
from typing import Dict, List, Tuple
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
        self.admm_model_instances: Dict[int, pyo.ConcreteModel] = {}

        self.admm_obj: float = 0.0

        # ADMM artifacts
        self.admm_z: Dict[int, float] = {}
        self.admm_λ: Dict[tuple, float] = {}

    def instantaniate_model(self, grid_data_parameters_dict: dict | None) -> None:
        """Instantiate the ADMM model with the provided grid data parameters."""
        if grid_data_parameters_dict is None:
            raise ValueError("Grid data parameters dictionary cannot be None.")
        self.admm_linear_model_instance = self.admm_linear_model.create_instance(grid_data_parameters_dict[list(grid_data_parameters_dict.keys())[0]])  # type: ignore
        for scen in grid_data_parameters_dict.keys():
            self.admm_model_instances[scen] = self.admm_model.create_instance(grid_data_parameters_dict[scen])  # type: ignore

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
        groups: Dict[int, List[int]] | int = 1,
        mutation_factor: int = 1,
    ) -> None:
        """Solve the ADMM model with the given parameters."""

        random.seed(seed_number)

        Ω = list(self.admm_model_instances.keys())  # type: ignore
        self.number_of_groups = len(groups) if isinstance(groups, dict) else groups

        # Sets
        switch_list = list(self.admm_model_instances[Ω[0]].S)  # type: ignore
        scenario_number = len(Ω)

        # Initialize consensus and duals
        z = {s: 0.5 for s in switch_list}  # consensus per switch
        λ = {
            (ω, s): 0.0
            for ω in itertools.product(Ω, range(self.number_of_groups))
            for s in switch_list
        }  # scaled duals
        results = self.solver.solve(
            self.admm_linear_model_instance, tee=self.config.verbose
        )
        δ_map = self.admm_linear_model_instance.δ.extract_values()  # type: ignore

        # Print initial δ_map state
        δ_values = ["█" if δ > 0.5 else "░" for δ in δ_map.values()]
        print(f"{''.join(δ_values)}")

        # ADMM iterations
        for k in range(1, max_iters + 1):
            # Broadcast z and λ into the model
            self._set_z_from_z(z)
            self._set_λ_from_λ(λ)
            # ---- x-update: solve each scenario with current z, λ ----
            δ_by_sc: Dict[Tuple[int, int], Dict[str, float]] = {}

            for ω in itertools.product(Ω, range(self.number_of_groups)):
                m = self.admm_model_instances[ω[0]]  # type: ignore
                # Solve multi‑scenario instance

                if isinstance(groups, int):
                    random_switches = random.sample(
                        switch_list,
                        k=min(
                            max(2, int(len(switch_list) / groups)) * mutation_factor,
                            len(switch_list) - 1,
                        ),
                    )
                else:
                    random_switches = []
                for edge_id, δ in δ_map.items():
                    if ((edge_id in random_switches) and isinstance(groups, int)) | (
                        (not isinstance(groups, int)) and (edge_id in groups[ω[1]])
                    ):
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
                        log.error(
                            f"[ADMM {k}] solve failed: {results.solver.termination_condition}"
                        )
                    δ_map = m.δ.extract_values()  # type: ignore
                    if None in δ_map.values():
                        raise ValueError(
                            f"[ADMM {k}] δ_map contains None values: {δ_map}"
                        )
                    δ_by_sc[ω] = δ_map

                    # Print δ_map state for current scenario with selection info
                    combined_values = []
                    for i, (switch_id, δ) in enumerate(
                        itertools.product(switch_list, δ_map.values())
                    ):
                        is_selected = switch_id in random_switches
                        is_closed = δ > 0.5

                        if is_selected and is_closed:
                            # Selected & Closed
                            combined_values.append("█")
                        elif is_selected and not is_closed:
                            # Selected & Open
                            combined_values.append("▓")
                        elif not is_selected and is_closed:
                            # Unselected & Closed
                            combined_values.append("▒")
                        else:
                            # Unselected & Open
                            combined_values.append("░")
                    print(f"{''.join(combined_values)}\n\n")

                except Exception as e:
                    log.error(
                        f"[ADMM {k}] Error extracting δ values for scenario {ω}: {e}"
                    )

            # ---- z-update (per switch) ----
            z_old = z.copy()
            for s in switch_list:
                z[s] = (
                    sum(
                        δ_by_sc[ω][s]
                        for ω in itertools.product(Ω, range(self.number_of_groups))
                    )
                ) / scenario_number

            # Print consensus z values
            z_values = ["█" if z[s] > 0.5 else "░" for s in switch_list]
            print(f"Iter {k}, Consensus z: {''.join(z_values)}")

            # ---- λ-update (scaled duals) ----
            for ω in itertools.product(Ω, range(self.number_of_groups)):
                for s in switch_list:
                    λ[(ω, s)] = λ[(ω, s)] + κ * (δ_by_sc[ω][s] - z[s])

            # ---- residuals (after all scenarios solved) ----
            r_sq = max(
                [
                    (δ_by_sc[ω][s] - z[s]) ** 2
                    for ω in itertools.product(Ω, range(self.number_of_groups))
                    for s in switch_list
                ]
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
        for ω in Ω:
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
            for g in range(self.number_of_groups):
                λ_param = getattr(m, "λ")
                for s in getattr(m, "S"):
                    λ_param[s].set_value(λ_map[((ω, g), s)])
