import itertools
import time
from typing import Dict, List, Tuple
import random
import pyomo.environ as pyo
import numpy as np
import polars as pl
from polars import col as c
from helpers import generate_log
from pipelines.reconfiguration.data_manager import PipelineDataManager
from data_model.reconfiguration_configs import ADMMConfig
from model_reconfiguration import generate_combined_model, generate_combined_lin_model
from pipelines.reconfiguration.model_managers import PipelineModelManager

log = generate_log(name=__name__)


class PipelineModelManagerADMM(PipelineModelManager):
    def __init__(self, config: ADMMConfig, data_manager: PipelineDataManager) -> None:
        """Initialize the combined model manager with configuration and data manager."""
        super().__init__(config, data_manager)
        self.config = config

        self.admm_model: pyo.AbstractModel = generate_combined_model()
        self.admm_linear_model: pyo.AbstractModel = generate_combined_lin_model()
        self.admm_linear_model_instance: pyo.ConcreteModel
        self.admm_model_instances: Dict[int, pyo.ConcreteModel] = {}

        # ADMM artifacts
        self.z: Dict[int, float] = {}
        self.λ: Dict[tuple, float] = {}
        self.zδ_variable: pl.DataFrame
        self.zζ_variable: pl.DataFrame
        self.s_norm_list: List[float] = []
        self.r_norm_list: List[float] = []
        self.time_list: List[float] = []

    def instantaniate_model(self, grid_data_parameters_dict: dict | None) -> None:
        """Instantiate the ADMM model with the provided grid data parameters."""
        if grid_data_parameters_dict is None:
            raise ValueError("Grid data parameters dictionary cannot be None.")
        self.admm_linear_model_instance = self.admm_linear_model.create_instance(grid_data_parameters_dict[list(grid_data_parameters_dict.keys())[0]])  # type: ignore
        for ω in grid_data_parameters_dict.keys():
            self.admm_model_instances[ω] = self.admm_model.create_instance(grid_data_parameters_dict[ω])  # type: ignore

    def solve_model(
        self, fixed_switches: bool = False, extract_duals: bool = False, **kwargs
    ):
        """Solve the ADMM model with the given parameters."""
        random.seed(self.config.seed_number)
        self.s_norm: float | None = None
        self.r_norm: float | None = None

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
        self.tr_taps = list(self.admm_model_instances[self.Ω[0]].TrTaps)  # type: ignore

        # Initialize consensus and duals
        self.zδ = {s: 0.5 for s in self.switch_list}  # consensus per switch
        self.λδ = {
            (ω, s): 0.0 for ω in self.Ω for s in self.switch_list
        }  # scaled duals

        self.zζ = {(tr, tap): 1.0 for tr, tap in self.tr_taps}
        self.λζ = {(ω, tr, tap): 0.0 for ω in self.Ω for tr, tap in self.tr_taps}

        δ_map, ζ_map = self.__solve_model(self.admm_linear_model_instance)

        self.time_list.append(time.process_time())
        # ADMM iterations
        for k in range(1, self.config.max_iters + 1 if not fixed_switches else 2):
            zδ_old = self.zδ.copy()
            zζ_old = self.zζ.copy()
            # Broadcast z and λ into the model
            self._set_zδ_from_zδ()
            self._set_λδ_from_λδ()
            self._set_zζ_from_zζ()
            self._set_λζ_from_λζ()
            # ---- x-update: solve each scenario with current z, λ ----
            δ_by_sc: Dict[int, Dict[str, float]] = {ω: δ_map for ω in self.Ω}
            ζ_by_sc: Dict[int, Dict[Tuple[str, str], float]] = {
                ω: ζ_map for ω in self.Ω
            }
            δ_by_sc, ζ_by_sc, δ_map, ζ_map = self.solve_admm_inner_loop(
                δ_map=δ_map,
                ζ_map=ζ_map,
                δ_by_sc=δ_by_sc,
                ζ_by_sc=ζ_by_sc,
                zδ_old=zδ_old,
                zζ_old=zζ_old,
                fixed_switches=fixed_switches,
                k=k,
            )

            # ---- λ-update (scaled duals) ----
            for ω, s in itertools.product(self.Ω, self.switch_list):
                self.λδ[(ω, s)] = self.λδ[(ω, s)] + self.κ * (
                    δ_by_sc[ω][s] - self.zδ[s]
                )
            for ω, (tr, tap) in itertools.product(self.Ω, self.zζ.keys()):
                self.λζ[(ω, tr, tap)] = self.λζ[(ω, tr, tap)] + self.κ * (
                    ζ_by_sc[ω][(tr, tap)] - self.zζ[(tr, tap)]
                )

            # ---- convergence ----
            if self.s_norm is None or self.r_norm is None:
                raise ValueError("Residuals s_norm and r_norm must be initialized.")
            if (self.r_norm <= self.config.ε_primal) and (
                self.s_norm <= self.config.ε_dual
            ):
                log.info(f"ADMM converged in {k} iterations.")
                break

            # ---- residual balancing (scaled ADMM) ----
            if self.r_norm > self.μ * self.s_norm:
                self.config.ρ *= self.τ_incr
            elif self.s_norm > self.μ * self.r_norm:
                self.config.ρ /= self.τ_decr
            for m in self.admm_model_instances.values():
                getattr(m, "ρ").set_value(self.config.ρ)

        # Refresh δ table after final iterate
        self.__get_δ_map()
        self.__get_ζ_map()
        self.__set_zδ_map()
        self.__set_zζ_map()

        if extract_duals:
            self.extract_dual_variables(
                δ_map=δ_map,
                ζ_map=ζ_map,
                δ_by_sc=δ_by_sc,
                ζ_by_sc=ζ_by_sc,
                zδ_old=zδ_old,
                zζ_old=zζ_old,
            )

        # store total objective (sum over scenarios), if available
        final_obj = float(sum(pyo.value(getattr(m, "objective")) for m in self.admm_model_instances.values()))  # type: ignore
        log.info(f"ADMM final objective (sum over scenarios) = {final_obj:.4f}")

    # ---------------------------
    # ADMM helpers and main loop
    # ---------------------------

    def extract_dual_variables(
        self,
        δ_map: Dict[str, float],
        ζ_map: Dict[Tuple[str, str], float],
        δ_by_sc: Dict[int, Dict[str, float]],
        ζ_by_sc: Dict[int, Dict[Tuple[str, str], float]],
        zδ_old: Dict[int, float],
        zζ_old: Dict[Tuple[str, str], float],
    ):
        for ω, m in self.admm_model_instances.items():
            m = self.__fix_taps(m, ζ_by_sc[ω])
            self.admm_model_instances[ω] = m

        self.solver = pyo.SolverFactory(self.config.solver_name)

        self.solve_admm_inner_loop(
            δ_map=δ_map,
            ζ_map=ζ_map,
            δ_by_sc=δ_by_sc,
            ζ_by_sc=ζ_by_sc,
            zδ_old=zδ_old,
            zζ_old=zζ_old,
            fixed_switches=True,
        )

    def solve_admm_inner_loop(
        self,
        δ_map: Dict[str, float],
        ζ_map: Dict[Tuple[str, str], float],
        δ_by_sc: Dict[int, Dict[str, float]],
        ζ_by_sc: Dict[int, Dict[Tuple[str, str], float]],
        zδ_old: Dict[int, float],
        zζ_old: Dict[Tuple[str, str], float],
        fixed_switches: bool,
        k: int | None = None,
    ) -> Tuple[
        Dict[int, Dict[str, float]],
        Dict[int, Dict[Tuple[str, str], float]],
        Dict[str, float],
        Dict[Tuple[str, str], float],
    ]:
        for ω, g in itertools.product(
            self.Ω, range(self.number_of_groups) if not fixed_switches else [None]
        ):
            m = self.admm_model_instances[ω]  # type: ignore
            selected_switches = self.__fix_switches(m, δ_map, g)

            δ_map, ζ_map = self.__solve_model(m, δ_map, ζ_map, selected_switches, k=k)
            δ_by_sc[ω] = δ_map
            ζ_by_sc[ω] = ζ_map

            # ---- z-update (per switch) ----
            for s in self.switch_list:
                self.zδ[s] = (sum(δ_by_sc[ω][s] for ω in self.Ω)) / len(self.Ω)
            # ---- zζ-update (per transformer tap) ----
            for tr, tap in self.zζ.keys():
                self.zζ[(tr, tap)] = sum(ζ_by_sc[ω][(tr, tap)] for ω in self.Ω) / len(
                    self.Ω
                )

            # ---- residuals (after all scenarios solved) ----
            self.r_norm, self.s_norm = self.__calculate_residuals(
                δ_by_sc=δ_by_sc,
                zδ=self.zδ,
                zδ_old=zδ_old,
                ζ_by_sc=ζ_by_sc,
                zζ=self.zζ,
                zζ_old=zζ_old,
            )
            self.r_norm_list.append(self.r_norm)
            self.s_norm_list.append(self.s_norm)
            self.time_list.append(time.process_time())
        return δ_by_sc, ζ_by_sc, δ_map, ζ_map

    def _set_zδ_from_zδ(self) -> None:
        """Broadcast consensus zδ[s] to δ_param[sc, s] for all scenarios sc."""
        for _, m in self.admm_model_instances.items():
            zδ_param = getattr(m, "zδ")
            for s in getattr(m, "S"):
                zδ_param[s].set_value(self.zδ[s])

    def _set_zζ_from_zζ(self) -> None:
        """Broadcast consensus zζ[tr, tap] to ζ_param[sc, (tr, tap)] for all scenarios sc."""
        for _, m in self.admm_model_instances.items():
            zζ_param = getattr(m, "zζ")
            for tr, tap in getattr(m, "TrTaps"):
                zζ_param[(tr, tap)].set_value(self.zζ[(tr, tap)])

    def _set_λδ_from_λδ(self) -> None:
        """Set per-scenario scaled duals λ_sc[s] from λ_map[(scen_id, s)]."""
        for ω, m in self.admm_model_instances.items():
            λδ_param = getattr(m, "λδ")
            for s in getattr(m, "S"):
                λδ_param[s].set_value(self.λδ[(ω, s)])

    def _set_λζ_from_λζ(self) -> None:
        """Set per-scenario scaled duals λζ_sc[(tr, tap)] from λζ_map[(scen_id, (tr, tap))]."""
        for ω, m in self.admm_model_instances.items():
            λζ_param = getattr(m, "λζ")
            for tr, tap in getattr(m, "TrTaps"):
                λζ_param[(tr, tap)].set_value(self.λζ[(ω, tr, tap)])

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

    def __get_ζ_map(self) -> None:
        """Extract the ζδ values from the model."""
        rows = []
        for ω in self.Ω:
            m = self.admm_model_instances[ω]
            ζ_map = m.ζ.extract_values()  # type: ignore
            for tr, tap in self.zζ.keys():
                rows.append((ω, tr, tap, float(ζ_map[(tr, tap)])))
        self.ζ_variable = pl.DataFrame(
            rows,
            schema=["SCEN", "TR", "TAP", "ζ_variable"],
            orient="row",
        )

    def __set_zδ_map(self) -> None:
        """Set the z values in the model from the z_map."""

        switches = self.data_manager.edge_data.filter(
            pl.col("type") == "switch"
        ).select("eq_fk", "edge_id", "normal_open")

        zδ_df = pl.DataFrame(
            {
                "edge_id": list(self.zδ.keys()),
                "zδ": list(self.zδ.values()),
            }
        )

        self.zδ_variable = (
            switches.join(zδ_df, on="edge_id", how="inner")
            .with_columns(
                (pl.col("zδ") > 0.5).alias("closed"),
                (~(pl.col("zδ") > 0.5)).alias("open"),
            )
            .select("eq_fk", "edge_id", "zδ", "normal_open", "closed", "open")
            .sort("edge_id")
        )

    def __set_zζ_map(self) -> None:
        """Set the zζ values in the model from the zζ_map."""
        rows = []
        for (tr, tap), zζ_value in self.zζ.items():
            rows.append((tr, tap, zζ_value))
        zζ_d = pl.DataFrame(
            rows,
            schema=["edge_id", "TAP", "zζ"],
            orient="row",
        )

        zζ_d = (
            self.data_manager.edge_data.filter(pl.col("type") == "transformer")
            .select("eq_fk", "edge_id", "u_of_edge", "v_of_edge")
            .join(zζ_d, on="edge_id", how="inner")
            .with_columns(
                (pl.col("zζ") > 0.5).alias("closed"),
                (~(pl.col("zζ") > 0.5)).alias("open"),
            )
            .select(
                "eq_fk",
                "edge_id",
                "u_of_edge",
                "v_of_edge",
                "TAP",
                "zζ",
                "closed",
                "open",
            )
            .sort("edge_id")
        )

        self.zζ_variable = (
            zζ_d.select(c("eq_fk"), c("u_of_edge"), c("v_of_edge"), c("TAP") * c("zζ"))
            .group_by("eq_fk")
            .sum()
        ).select(
            c("eq_fk").alias("eq_fk"),
            c("u_of_edge").alias("u_of_edge"),
            c("v_of_edge").alias("v_of_edge"),
            c("TAP").alias("tap_value"),
        )

    def __print_switch_states(
        self,
        δ_map: dict,
        k: int | None = None,
        selected_switches: list | None = None,
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
        log.info(
            f"{''.join(combined_values)}"
            + (f" [ADMM {k}]" if k is not None else "")
            + (
                f" r_norm={self.r_norm:.3f}, s_norm={self.s_norm:.3f}, ρ={self.config.ρ:.3f}"
                if self.s_norm is not None and self.r_norm is not None
                else ""
            )
        )

    def __fix_switches(self, m: pyo.ConcreteModel, δ_map: dict, g: int | None) -> List:
        """Fix switches based on δ_map and group g."""
        if isinstance(self.groups, int) and (g is not None):
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
                (not isinstance(self.groups, int) and (g is not None))
                and (edge_id in self.groups[g])
            ):
                m.δ[edge_id].unfix()  # type: ignore
            else:
                m.δ[edge_id].fix(1.0 if δ > 0.5 else 0.0)  # type: ignore

        return (
            random_switches
            if isinstance(self.groups, int) or g is None
            else self.groups[g]
        )

    def __fix_taps(
        self,
        m: pyo.ConcreteModel,
        ζ_map: Dict[Tuple[str, str], float],
    ) -> pyo.ConcreteModel:
        for (tr, tap), ζ in ζ_map.items():
            m.ζ[(tr, tap)].fix(ζ)  # type: ignore
        return m

    def __solve_model(
        self,
        model: pyo.ConcreteModel,
        δ_map: Dict[str, float] = {},
        ζ_map: Dict[Tuple[str, str], float] = {},
        selected_switches: List | None = None,
        k: int | None = None,
    ) -> Tuple[Dict[str, float], Dict[Tuple[str, str], float]]:
        """Solve the given model and return the δ values."""
        try:
            results = self.solver.solve(model, tee=self.config.verbose)
            if results.solver.termination_condition != pyo.TerminationCondition.optimal:
                log.error(f"Model solve failed: {results.solver.termination_condition}")
            δ_map = model.δ.extract_values()  # type: ignore
            ζ_map = model.ζ.extract_values()  # type: ignore
            if None in δ_map.values():
                raise ValueError(f"δ_map contains None values: {δ_map}")
            self.__print_switch_states(δ_map, k=k, selected_switches=selected_switches)
        except Exception as e:
            log.error(f"Error solving model: {e}")
        return δ_map, ζ_map

    def __calculate_residuals(
        self,
        δ_by_sc: Dict[int, Dict[str, float]],
        zδ: Dict[int, float],
        zδ_old: Dict[int, float],
        ζ_by_sc: Dict[int, Dict[Tuple[str, str], float]],
        zζ: Dict[Tuple[str, str], float],
        zζ_old: Dict[Tuple[str, str], float],
    ) -> Tuple[float, float]:
        """Calculate the primal and dual residuals."""
        r_norm1 = float(
            np.sqrt(
                max(
                    [
                        (δ_by_sc[ω][s] - zδ[s]) ** 2
                        for ω in self.Ω
                        for s in self.switch_list
                    ]
                )
            )
        )

        s_norm1 = float(
            self.config.ρ
            * np.sqrt(sum((zδ[s] - zδ_old[s]) ** 2 for s in self.switch_list))
        )
        r_norm2 = float(
            np.sqrt(
                max(
                    [
                        (ζ_by_sc[ω][(tr, tap)] - zζ[(tr, tap)]) ** 2
                        for ω in self.Ω
                        for (tr, tap) in self.zζ.keys()
                    ]
                )
            )
        )

        s_norm2 = float(
            self.config.ρ
            * np.sqrt(
                sum(
                    (zζ[(tr, tap)] - zζ_old[(tr, tap)]) ** 2
                    for (tr, tap) in self.zζ.keys()
                )
            )
        )

        r_norm = max(r_norm1, r_norm2)
        s_norm = s_norm1 + s_norm2

        return r_norm, s_norm
