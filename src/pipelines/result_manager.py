import polars as pl
from polars import col as c
from pipelines.data_manager import PipelineDataManager
from pipelines.model_managers.combined import PipelineModelManagerCombined
from pipelines.model_managers.bender import PipelineModelManagerBender
from polars_function import cast_boolean
from pyomo_utility import extract_optimization_results


class PipelineResultManager:
    """
    A class to manage the results of a pipeline, including data processing and visualization.
    """

    def __init__(
        self,
        data_manager: PipelineDataManager,
        model_manager: PipelineModelManagerCombined | PipelineModelManagerBender,
    ):
        """
        Initialize the result manager with a data manager and a model manager.
        """
        self.data_manager = data_manager
        self.model_manager = model_manager

    def init_model_instance(self):
        if isinstance(self.model_manager, PipelineModelManagerCombined):
            self.model_instance = self.model_manager.combined_model_instance
        elif isinstance(self.model_manager, PipelineModelManagerBender):
            self.model_instance = self.model_manager.optimal_slave_model_instance

    def extract_switch_status(self) -> pl.DataFrame:
        self.init_model_instance()
        # Pull out only the switch edges...
        ss = self.data_manager.edge_data.filter(c("type") == "switch")
        # Build a Polars mapping from edge_id -> {0,1}
        delta_map = self.model_instance.delta.extract_values()  # type: ignore
        # 1) Create a 'delta' column exactly as before
        ss = ss.with_columns(
            c("edge_id")
            .replace_strict(delta_map, default=None)  # None for missing entries
            .alias("delta")  # still 0,1 or None
        )
        # 2) Create 'open' by bitwise‐not on the Boolean view of delta
        ss = ss.with_columns((~(c("delta") > 0.5).pipe(cast_boolean)).alias("open"))
        # Finally select the columns you care about
        return ss.select(["eq_fk", "edge_id", "delta", "normal_open", "open"])

    def extract_node_voltage(self) -> pl.DataFrame:
        self.init_model_instance()

        return (
            extract_optimization_results(self.model_instance, "v_sq")
            .select((c("v_sq")).sqrt().alias("v_pu"), c("N").alias("node_id"))
            .join(
                self.data_manager.node_data[["cn_fk", "node_id", "v_base"]],
                on="node_id",
                how="left",
            )
        )

    def extract_edge_current(self) -> pl.DataFrame:
        self.init_model_instance()
        return (
            extract_optimization_results(self.model_instance, "i_sq")
            .select(
                (c("i_sq")).sqrt().alias("i_pu"), c("C").list.get(0).alias("edge_id")
            )
            .group_by("edge_id")
            .agg(c("i_pu").max())
            .sort("edge_id")
            .join(
                self.data_manager.edge_data.filter(c("type") != "switch")[
                    ["eq_fk", "edge_id", "i_base"]
                ],
                on="edge_id",
                how="inner",
            )
        )

    def extract_edge_active_power_flow(self) -> pl.DataFrame:
        self.init_model_instance()
        return (
            extract_optimization_results(self.model_instance, "p_flow")
            .select(
                c("p_flow").alias("p_pu"),
                c("C").list.get(0).alias("edge_id"),
                c("C").list.get(1).alias("from_node_id"),
                c("C").list.get(2).alias("to_node_id"),
            )
            .join(
                self.data_manager.edge_data.filter(c("type") != "switch")[
                    ["eq_fk", "edge_id"]
                ],
                on="edge_id",
                how="inner",
            )
        )

    def extract_edge_reactive_power_flow(self) -> pl.DataFrame:
        self.init_model_instance()
        return (
            extract_optimization_results(self.model_instance, "q_flow")
            .select(
                c("q_flow").alias("q_pu"),
                c("C").list.get(0).alias("edge_id"),
                c("C").list.get(1).alias("from_node_id"),
                c("C").list.get(2).alias("to_node_id"),
            )
            .join(
                self.data_manager.edge_data.filter(c("type") != "switch")[
                    ["eq_fk", "edge_id"]
                ],
                on="edge_id",
                how="inner",
            )
        )
