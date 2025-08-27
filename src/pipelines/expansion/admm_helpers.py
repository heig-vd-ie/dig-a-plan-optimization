import json
from typing import Dict, List
from pathlib import Path
from dataclasses import dataclass
import polars as pl
from polars import col as c
import patito as pt
from data_schema import NodeEdgeModel, NodeData
from pipelines.reconfiguration import DigAPlanADMM
from pipelines.reconfiguration.configs import ADMMConfig, PipelineType


@dataclass
class ADMMResult:
    duals: pl.DataFrame
    θs: pl.DataFrame
    load0: pl.DataFrame
    pv0: pl.DataFrame
    cap0: pl.DataFrame
    results: Dict


class ADMM:
    """ADMM (Alternating Direction Method of Multipliers) optimization class."""

    def __init__(
        self,
        groups: Dict[int, List[int]] | int,
        grid_data: NodeEdgeModel,
        solver_non_convex: int,
        time_limit: int,
        big_m: float = 1e3,
        ε: float = 1e-4,
        ρ: float = 2.0,
        γ_infeasibility: float = 1.0,
        γ_admm_penalty: float = 1.0,
        γ_trafo_loss: float = 1e2,
        max_iters: int = 10,
        μ: float = 10.0,
        τ_incr: float = 2.0,
        τ_decr: float = 2.0,
    ):
        self.config = ADMMConfig(
            verbose=False,
            pipeline_type=PipelineType.ADMM,
            solver_name="gurobi",
            solver_non_convex=solver_non_convex,
            big_m=big_m,
            ε=ε,
            ρ=ρ,
            γ_infeasibility=γ_infeasibility,
            γ_admm_penalty=γ_admm_penalty,
            γ_trafo_loss=γ_trafo_loss,
            max_iters=max_iters,
            μ=μ,
            τ_incr=τ_incr,
            τ_decr=τ_decr,
            time_limit=time_limit,
            groups=groups,
        )
        self.grid_data = grid_data

    def update_node_grid_data(
        self,
        δ_load: List[float],
        δ_pv: List[float],
        node_ids: List[int],
    ):
        updates_df = pl.DataFrame(
            {
                "node_id": node_ids,
                "new_cons_installed": δ_load,
                "new_prod_installed": δ_pv,
            }
        )

        updated_node_data = (
            self.grid_data.node_data.as_polars()
            .join(updates_df, on="node_id", how="left")
            .with_columns(
                [
                    pl.sum_horizontal(
                        [c("new_cons_installed").fill_null(0.0), c("cons_installed")]
                    ).alias("cons_installed"),
                    pl.sum_horizontal(
                        [c("new_prod_installed").fill_null(0.0), c("prod_installed")]
                    ).alias("prod_installed"),
                ]
            )
            .drop(["new_cons_installed", "new_prod_installed"])
        )

        self.grid_data.node_data = (
            pt.DataFrame(updated_node_data).set_model(NodeData).cast(strict=True)
        )

    def update_edge_grid_data(self, δ_cap: List[float], edge_ids: List[int]):
        updates_df = pl.DataFrame(
            {
                "edge_id": edge_ids,
                "new_capacity": δ_cap,
            }
        )

        updated_edge_data = (
            self.grid_data.edge_data.as_polars()
            .join(updates_df, on="edge_id", how="left")
            .with_columns(
                [
                    pl.sum_horizontal(
                        [c("new_capacity").fill_null(0.0), c("i_max_pu")]
                    ).alias("i_max_pu"),
                    pl.sum_horizontal(
                        [c("new_capacity").fill_null(0.0), c("p_max_pu")]
                    ).alias("p_max_pu"),
                ]
            )
            .drop(["new_capacity"])
        )

        self.grid_data.edge_data = (
            pt.DataFrame(updated_edge_data).set_model(NodeData).cast(strict=True)
        )

    def update_grid_data(
        self,
        δ_load: List[float],
        δ_pv: List[float],
        node_ids: List[int],
        δ_cap: List[float],
        edge_ids: List[int],
    ):
        self.update_node_grid_data(δ_load, δ_pv, node_ids)
        self.update_edge_grid_data(δ_cap, edge_ids)

    def _record_results(self) -> Dict:
        """Record the results of the optimization."""
        results = {
            "voltage": {},
            "current": {},
            "real_power": {},
            "reactive_power": {},
            "switches": self.dap.result_manager.extract_switch_status().to_dicts(),
            "taps": self.dap.result_manager.extract_transformer_tap_position().to_dicts(),
            "r_norm": self.dap.model_manager.r_norm_list,
            "s_norm": self.dap.model_manager.s_norm_list,
        }

        if self.dap.data_manager.grid_data_parameters_dict is None:
            raise ValueError("No grid data parameters found.")

        for ω in range(
            len(list(self.dap.data_manager.grid_data_parameters_dict.keys()))
        ):
            results["voltage"][ω] = self.dap.result_manager.extract_node_voltage(
                scenario=ω
            ).to_dicts()
            results["current"][ω] = self.dap.result_manager.extract_edge_current(
                scenario=ω
            ).to_dicts()
            results["real_power"][ω] = (
                self.dap.result_manager.extract_edge_active_power_flow(
                    scenario=ω
                ).to_dicts()
            )
            results["reactive_power"][ω] = (
                self.dap.result_manager.extract_edge_reactive_power_flow(
                    scenario=ω
                ).to_dicts()
            )

        return results

    def solve(self) -> ADMMResult:
        """Solve the optimization problem using ADMM."""
        self.dap = DigAPlanADMM(config=self.config)
        self.dap.add_grid_data(grid_data=self.grid_data)
        self.dap.solve_model(extract_duals=True)
        duals = self.dap.result_manager.extract_duals_for_expansion()
        θs = self.dap.result_manager.extract_reconfiguration_θ()
        load0 = self.dap.data_manager.node_data[["node_id", "cons_installed"]]
        pv0 = self.dap.data_manager.node_data[["node_id", "prod_installed"]]
        cap0 = self.dap.data_manager.edge_data[["edge_id", "p_max_pu"]]
        results = self._record_results()
        return ADMMResult(
            duals=duals,
            θs=θs,
            load0=load0,
            pv0=pv0,
            cap0=cap0,
            results=results,
        )
