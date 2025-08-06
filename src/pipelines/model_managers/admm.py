import itertools
from typing import Dict, List
import random
import pyomo.environ as pyo
import numpy as np
import polars as pl
from general_function import generate_log
from pipelines.data_manager import PipelineDataManager
from pipelines.configs import ADMMConfig, PipelineType
from optimization_model import generate_combined_model, generate_combined_lin_model
from pipelines.model_managers import PipelineModelManager

log = generate_log(name=__name__)


class PipelineModelManagerADMM(PipelineModelManager):
    def __init__(self, config: ADMMConfig, data_manager: PipelineDataManager) -> None:
        """Initialize the combined model manager with configuration and data manager."""
        super().__init__(config, data_manager, PipelineType.ADMM)
        self.config = config

        self.admm_model: pyo.AbstractModel = generate_combined_model()
        self.admm_linear_model: pyo.AbstractModel = generate_combined_lin_model()
        self.admm_linear_model_instance: pyo.ConcreteModel
        self.admm_model_instances: Dict[int, pyo.ConcreteModel] = {}

        # ADMM artifacts
        self.z: Dict[int, float] = {}
        self.λ: Dict[tuple, float] = {}

    def instantaniate_model(self, grid_data_parameters_dict: dict | None) -> None:
        """Instantiate the ADMM model with the provided grid data parameters."""
        if grid_data_parameters_dict is None:
            raise ValueError("Grid data parameters dictionary cannot be None.")
        self.admm_linear_model_instance = self.admm_linear_model.create_instance(grid_data_parameters_dict[list(grid_data_parameters_dict.keys())[0]])  # type: ignore
        for scen in grid_data_parameters_dict.keys():
            self.admm_model_instances[scen] = self.admm_model.create_instance(grid_data_parameters_dict[scen])  # type: ignore

    def solve_model(self) -> None:
        """Solve the ADMM model with the given parameters."""
        random.seed(self.config.seed_number)

        self.mutation_factor = self.config.mutation_factor
        self.groups = self.config.groups
        self.κ = self.config.κ
        self.μ = self.config.μ
        self.τ_incr = self.config.τ_incr
        self.τ_decr = self.config.τ_decr

        self.Ω = list(self.admm_model_instances.keys())  # type: ignore
        self.number_of_groups = (
            len(self.config.groups)
            if isinstance(self.config.groups, dict)
            else self.config.groups
        )

        # Sets
        self.switch_list = list(self.admm_model_instances[self.Ω[0]].S)  # type: ignore
        scenario_number = len(self.Ω)

        # Initialize consensus and duals
        self.z = {s: 0.5 for s in self.switch_list}  # consensus per switch
        self.λ = {(ω, s): 0.0 for ω in self.Ω for s in self.switch_list}  # scaled duals

        δ_map = self.__solve_model(self.admm_linear_model_instance)
        # ADMM iterations
        for k in range(1, self.config.max_iters + 1):
            # Broadcast z and λ into the model
            self._set_z_from_z()
            self._set_λ_from_λ()
            # ---- x-update: solve each scenario with current z, λ ----
            δ_by_sc: Dict[int, Dict[str, float]] = {}

            for ω, g in itertools.product(self.Ω, range(self.number_of_groups)):
                m = self.admm_model_instances[ω]  # type: ignore
                selected_switches = self.__fix_switches(m, δ_map, g)
                δ_map = self.__solve_model(m, δ_map, selected_switches)
                δ_by_sc[ω] = δ_map

            # ---- z-update (per switch) ----
            z_old = self.z.copy()
            for s in self.switch_list:
                self.z[s] = (sum(δ_by_sc[ω][s] for ω in self.Ω)) / scenario_number

            # Print consensus z values
            self.__print_switch_states(self.z)

            # ---- λ-update (scaled duals) ----
            for ω, s in itertools.product(self.Ω, self.switch_list):
                self.λ[(ω, s)] = self.λ[(ω, s)] + self.κ * (δ_by_sc[ω][s] - self.z[s])

            # ---- residuals (after all scenarios solved) ----
            r_norm, s_norm = self.__calculate_residuals(δ_by_sc, self.z, z_old, k)
            # ---- convergence ----
            if (r_norm <= self.config.ε_primal) and (s_norm <= self.config.ε_dual):
                log.info(f"ADMM converged in {k} iterations.")
                break

        # Refresh δ table after final iterate
        self.__get_δ_map()

        # store total objective (sum over scenarios), if available
        final_obj = float(sum(pyo.value(getattr(m, "objective")) for m in self.admm_model_instances.values()))  # type: ignore
        log.info(f"ADMM final objective (sum over scenarios) = {final_obj:.4f}")

    # ---------------------------
    # ADMM helpers and main loop
    # ---------------------------

    def _set_z_from_z(self) -> None:
        """Broadcast consensus z[s] to del_param[sc, s] for all scenarios sc."""
        for _, m in self.admm_model_instances.items():
            z_param = getattr(m, "z")
            for s in getattr(m, "S"):
                z_param[s].set_value(self.z[s])

    def _set_λ_from_λ(self) -> None:
        """Set per-scenario scaled duals λ_sc[s] from λ_map[(scen_id, s)]."""
        for ω, m in self.admm_model_instances.items():
            λ_param = getattr(m, "λ")
            for s in getattr(m, "S"):
                λ_param[s].set_value(self.λ[(ω, s)])

    def __get_δ_map(self) -> None:
        """Extract the δ values from the model."""
        rows = []
        for ω in self.Ω:
            m = self.admm_model_instances[ω]
            δ_map = m.δ.extract_values()  # type: ignore
            for s in self.switch_list:
                rows.append((ω, s, float(δ_map[s])))
        self.δ_variable = pl.DataFrame(
            rows,
            schema=["SCEN", "S", "δ_variable"],
            orient="row",
        )

    def __print_switch_states(
        self, δ_map: dict, selected_switches: list | None = None
    ) -> None:
        """Print the switch states based on δ_map."""
        combined_values = []
        for switch_id, δ in δ_map.items():
            is_selected = (
                switch_id in selected_switches
                if selected_switches is not None
                else False
            )
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
        print(f"{''.join(combined_values)}")

    def __fix_switches(self, m: pyo.ConcreteModel, δ_map: dict, g: int) -> List:
        """Fix switches based on δ_map and group g."""
        if isinstance(self.groups, int):
            random_switches = random.sample(
                self.switch_list,
                k=min(
                    max(
                        2,
                        int(len(self.switch_list) / self.groups * self.mutation_factor),
                    ),
                    len(self.switch_list) - 1,
                ),
            )
        else:
            random_switches = []
        for edge_id, δ in δ_map.items():
            if ((edge_id in random_switches) and isinstance(self.groups, int)) | (
                (not isinstance(self.groups, int)) and (edge_id in self.groups[g])
            ):
                m.δ[edge_id].unfix()  # type: ignore
            else:
                m.δ[edge_id].fix(1.0 if δ > 0.5 else 0.0)  # type: ignore

        return random_switches if isinstance(self.groups, int) else self.groups[g]

    def __solve_model(
        self,
        model: pyo.ConcreteModel,
        δ_map: Dict[str, float] = {},
        selected_switches: List | None = None,
    ) -> Dict[str, float]:
        """Solve the given model and return the δ values."""
        try:
            results = self.solver.solve(model, tee=self.config.verbose)
            if results.solver.termination_condition != pyo.TerminationCondition.optimal:
                log.error(f"Model solve failed: {results.solver.termination_condition}")
            δ_map = model.δ.extract_values()  # type: ignore
            if None in δ_map.values():
                raise ValueError(f"δ_map contains None values: {δ_map}")
            self.__print_switch_states(δ_map, selected_switches)
        except Exception as e:
            log.error(f"Error solving model: {e}")
        return δ_map

    def __calculate_residuals(
        self,
        δ_by_sc: Dict[int, Dict[str, float]],
        z: Dict[int, float],
        z_old: Dict[int, float],
        k: int,
    ) -> tuple[float, float]:
        """Calculate the primal and dual residuals."""
        r_norm = float(
            np.sqrt(
                max(
                    [
                        (δ_by_sc[ω][s] - z[s]) ** 2
                        for ω in self.Ω
                        for s in self.switch_list
                    ]
                )
            )
        )

        s_norm = float(
            self.config.ρ
            * np.sqrt(sum((z[s] - z_old[s]) ** 2 for s in self.switch_list))
        )

        log.info(f"[ADMM {k}] r={r_norm:.3e}, s={s_norm:.3e}, ρ={self.config.ρ:.3g}")

        # ---- residual balancing (scaled ADMM) ----
        if r_norm > self.μ * s_norm:
            self.config.ρ *= self.τ_incr
        elif s_norm > self.μ * r_norm:
            self.config.ρ /= self.τ_decr
        for m in self.admm_model_instances.values():
            getattr(m, "ρ").set_value(self.config.ρ)
        return r_norm, s_norm
