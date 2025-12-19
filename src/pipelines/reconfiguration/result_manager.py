from typing import List
import polars as pl
from polars import col as c
from pipelines.reconfiguration.data_manager import PipelineDataManager
from pipelines.reconfiguration.model_managers.combined import (
    PipelineModelManagerCombined,
)
from pipelines.reconfiguration.model_managers.bender import PipelineModelManagerBender
from pipelines.reconfiguration.model_managers.admm import PipelineModelManagerADMM
from helper_functions import cast_boolean, modify_string_col
from pipelines.helpers.pyomo_utility import extract_optimization_results


class PipelineResultManager:
    """
    A class to manage the results of a pipeline, including data processing and visualization.
    """

    def __init__(
        self,
        data_manager: PipelineDataManager,
        model_manager: (
            PipelineModelManagerCombined
            | PipelineModelManagerBender
            | PipelineModelManagerADMM
        ),
    ):
        """
        Initialize the result manager with a data manager and a model manager.
        """
        self.data_manager = data_manager
        self.model_manager = model_manager

    def init_model_instance(self, scenario: int = 0):
        if isinstance(self.model_manager, PipelineModelManagerADMM):
            self.model_instance = self.model_manager.admm_model_instances[
                list(self.model_manager.admm_model_instances.keys())[scenario]
            ]
        elif isinstance(self.model_manager, PipelineModelManagerCombined):
            self.model_instance = self.model_manager.combined_model_instance
        elif isinstance(self.model_manager, PipelineModelManagerBender):
            self.model_instance = self.model_manager.optimal_slave_model_instance

    def extract_switch_status(self) -> pl.DataFrame:
        self.init_model_instance()
        # Pull out only the switch edges...
        ss = self.data_manager.edge_data.filter(c("type") == "switch")
        # Build a Polars mapping from edge_id -> {0,1}
        δ_map = self.model_instance.δ.extract_values()  # type: ignore
        # 1) Create a 'δ' column exactly as before
        ss = ss.with_columns(
            c("edge_id")
            .replace_strict(δ_map, default=None)  # None for missing entries
            .alias("δ")  # still 0,1 or None
        )
        # 2) Create 'open' by bitwise‐not on the Boolean view of δ
        ss = ss.with_columns((~(c("δ") > 0.5).pipe(cast_boolean)).alias("open"))
        # Finally select the columns you care about
        return ss.select(["eq_fk", "edge_id", "δ", "normal_open", "open"])

    def extract_transformer_tap_position(self) -> pl.DataFrame:
        self.init_model_instance()
        tt = self.data_manager.edge_data.filter(c("type") == "transformer")
        # Extract tap positions from the model instance
        ζ_map = self.model_instance.ζ.extract_values()  # type: ignore

        rows = []
        for (tr, tap), zζ_value in ζ_map.items():
            rows.append((tr, tap, zζ_value))
        ζ_d = pl.DataFrame(
            rows,
            schema=["edge_id", "TAP", "ζ"],
            orient="row",
        )
        ζ = (
            ζ_d.with_columns(
                c("edge_id"),
                (c("TAP") * c("ζ")).alias("tap_value"),
            )
            .group_by("edge_id")
            .agg(c("tap_value").sum().alias("tap_value"))
        )

        # Create a Polars DataFrame with transformer tap positions
        tt = tt.join(
            ζ,
            on="edge_id",
        )
        return tt

    def extract_nodal_variables(self, variable: str, scenario: int = 0) -> pl.DataFrame:
        self.init_model_instance(scenario=scenario)

        return (
            extract_optimization_results(self.model_instance, variable)
            .select((c(variable)).sqrt(), c("NΩ").list.get(0).alias("node_id"))
            .join(
                self.data_manager.node_data[
                    ["cn_fk", "node_id", "v_base", "min_vm_pu", "max_vm_pu"]
                ],
                on="node_id",
                how="full",
            )
        )

    def extract_edge_variables(self, variable: str, scenario: int = 0) -> pl.DataFrame:
        self.init_model_instance(scenario=scenario)

        return (
            extract_optimization_results(self.model_instance, variable)
            .select(
                (c(variable)),
                c("CΩ").list.get(0).alias("edge_id"),
                c("CΩ").list.get(1).alias("from_node_id"),
                c("CΩ").list.get(2).alias("to_node_id"),
            )
            .join(
                self.data_manager.edge_data.filter(c("type") != "switch")[
                    ["eq_fk", "edge_id", "i_base", "type", "i_max_pu"]
                ],
                on="edge_id",
                how="full",
            )
        ).filter(c("type") != "switch")

    def extract_node_voltage(self, scenario: int = 0) -> pl.DataFrame:
        return self.extract_nodal_variables("v_sq", scenario=scenario).with_columns(
            (c("v_sq") ** 0.5).alias("v_pu")
        )

    def extract_edge_current(self, scenario: int = 0) -> pl.DataFrame:
        p = self.extract_edge_variables("p_flow", scenario=scenario)
        q = self.extract_edge_variables("q_flow", scenario=scenario)
        i = self.extract_edge_variables("i_sq", scenario=scenario)
        return (
            p.join(
                q[["edge_id", "from_node_id", "to_node_id", "q_flow"]],
                on=["edge_id", "from_node_id", "to_node_id"],
                how="inner",
            )
            .join(
                i[["edge_id", "from_node_id", "to_node_id", "i_sq"]],
                on=["edge_id", "from_node_id", "to_node_id"],
                how="inner",
            )
            .with_columns(
                [
                    (c("i_sq") ** 0.5).alias("i_pu"),
                    ((c("i_sq") ** 0.5) * 100 / c("i_max_pu")).alias("i_pct"),
                ]
            )
        )

    def extract_edge_active_power_flow(self, scenario: int = 0) -> pl.DataFrame:
        return self.extract_edge_variables("p_flow", scenario=scenario).with_columns(
            c("p_flow").alias("p_pu")
        )

    def extract_edge_reactive_power_flow(self, scenario: int = 0) -> pl.DataFrame:
        return self.extract_edge_variables("q_flow", scenario=scenario).with_columns(
            c("q_flow").alias("q_pu")
        )

    def extract_dual_variables(self, scenario: int = 0) -> pl.DataFrame:
        self.init_model_instance(scenario=scenario)
        return pl.DataFrame(
            {
                "name": list(dict(self.model_instance.dual).keys()),  # type: ignore
                "value": list(dict(self.model_instance.dual).values()),  # type: ignore
            }
        )

    def extract_duals_for_expansion(
        self, constraint_names: List[str] | None = None
    ) -> pl.DataFrame:
        """Extract dual variables for a specific constraint."""
        if constraint_names is None:
            constraint_names = [
                "current_limit",
                "current_limit_tr",
                "installed_cons",
                "installed_prod",
            ]
        if not isinstance(self.model_manager, PipelineModelManagerADMM):
            raise NotImplementedError("ADMM dual extraction is not implemented")
        duals = []
        for scenario, ω in enumerate(self.model_manager.Ω):
            dual = (
                self.extract_dual_variables(scenario=scenario)
                .with_columns(
                    c("name").map_elements(lambda x: x.name, return_dtype=pl.Utf8),
                    (pl.lit(ω).alias("ω")),
                )
                .drop_nulls(subset="value")
            )
            if not dual.is_empty():
                duals.append(dual)
        duals_df: pl.DataFrame = pl.concat(duals, how="vertical")
        duals_df = (
            duals_df.with_columns(
                c("name")
                .pipe(modify_string_col, format_str={"]": ""})
                .str.split("[")
                .list.to_struct(fields=["name", "index"])
            )
            .unnest("name")
            .filter(c("name").is_in(constraint_names))
        )
        duals_df = (
            duals_df.select(
                [
                    c("name"),
                    c("ω"),
                    c("index").str.split(",").list.get(0).alias("id"),
                    c("value"),
                ]
            )
            .group_by(["name", "id", "ω"])
            .agg(c("value").abs().sum().alias("value"))
        ).sort(["name", "id", "ω"])
        return duals_df

    def extract_reconfiguration_θ(self) -> pl.DataFrame:
        """Extract reconfiguration angles."""
        if not isinstance(self.model_manager, PipelineModelManagerADMM):
            raise NotImplementedError("Reconfiguration θ extraction is not implemented")
        θs = []
        for scenario, ω in enumerate(self.model_manager.Ω):
            self.init_model_instance(scenario=scenario)
            θs.append(
                extract_optimization_results(self.model_instance, "θ").select(
                    [pl.lit(ω).alias("ω"), c("θ")]
                )
            )
        return pl.concat(θs, how="vertical")
