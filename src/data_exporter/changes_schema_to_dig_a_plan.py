import polars as pl
import numpy as np
from polars import col as c
from data_schema import NodeEdgeModel
from data_schema import NodeData, EdgeData, LoadData
from twindigrid_changes.schema import ChangesSchema
from general_function import pl_to_dict
from data_exporter import validate_data


def change_schema_to_dig_a_plan_schema(
    change_schema: ChangesSchema,
    s_base: float = 1e6,
) -> NodeEdgeModel:

    grid_data: dict[str, pl.DataFrame] = {}

    connectivity: pl.DataFrame = change_schema.connectivity
    ext_grid_id = connectivity.filter(c("eq_class") == "external_network")["cn_fk"][0]

    node_data = change_schema.connectivity_node.with_row_index(
        name="node_id"
    ).with_columns(
        c("uuid").alias("cn_fk"),
        c("base_voltage_fk").alias("v_base"),
        (s_base / (c("base_voltage_fk") * np.sqrt(3))).alias("i_base"),
    )

    container_mapping = pl_to_dict(
        connectivity.filter(c("eq_class") == "energy_consumer").filter(
            c("container_fk").is_first_distinct()
        )["container_fk", "cn_fk"]
    )

    measurement_value_mapping = pl_to_dict(
        change_schema.measurement_span["measurement_fk", "double_value"].unique(
            "measurement_fk"
        )
    )
    measurement = (
        change_schema.measurement.filter(c("source_fk") == "conventional_meter")
        .with_columns(
            c("uuid")
            .replace_strict(measurement_value_mapping, default=None)
            .alias("value"),
            c("resource_fk")
            .replace_strict(container_mapping, default=None)
            .alias("cn_fk"),
        )
        .with_columns(
            (pl.lit(10).pow(c("unit_multiplier")) * c("value") / (24 * 365)).alias(
                "value"
            )
        )
        .group_by("cn_fk")
        .agg((c("value").sum() / s_base).alias("p_node_pu"))
    )

    grid_data["node_data"] = node_data.join(
        measurement, on="cn_fk", how="left"
    ).with_columns(
        pl.when(c("cn_fk") == ext_grid_id)
        .then(pl.lit("slack"))
        .otherwise(pl.lit("pq"))
        .alias("type"),
        pl.when(c("cn_fk") == ext_grid_id)
        .then(pl.lit(1.1))
        .otherwise(pl.lit(None))
        .alias("v_node_sqr_pu"),
    )

    node_mapping = pl_to_dict(node_data["cn_fk", "node_id"])

    eq_connectivity = (
        connectivity.with_columns(c("cn_fk").replace_strict(node_mapping, default=None))
        .pivot(on="side", values="cn_fk", index="eq_fk")
        .rename({"t1": "u_of_edge", "t2": "v_of_edge"})
    )

    branch_parameter_event = pl.DataFrame(change_schema.branch_parameter_event)
    branch = pl.DataFrame(change_schema.branch).join(
        branch_parameter_event, left_on="uuid", right_on="eq_fk", how="left"
    )

    branch = (
        branch.join(eq_connectivity, left_on="uuid", right_on="eq_fk", how="left")
        .join(
            node_data["node_id", "v_base", "i_base"],
            left_on="u_of_edge",
            right_on="node_id",
            how="left",
        )
        .with_columns(
            (c("v_base") ** 2 / s_base).alias("z_base"),
        )
        .select(
            "u_of_edge",
            "v_of_edge",
            c("uuid").alias("eq_fk"),
            (c("r") / c("z_base")).alias("r_pu"),
            (c("x") / c("z_base")).alias("x_pu"),
            (c("b") * c("z_base")).alias("b_pu"),
            (c("current_limit") / c("i_base")).alias("i_max_pu"),
            pl.lit("branch").alias("type"),
            c("i_base"),
            (np.sqrt(3) * c("current_limit") * c("v_base") / s_base).alias("p_max_pu"),
        )
    )

    transformer_end = change_schema.transformer_end.pivot(
        on="side", values="nominal_voltage", index="eq_fk"
    ).rename({"t1": "vn_hv", "t2": "vn_lv"})

    transformer = (
        pl.DataFrame(change_schema.transformer)
        .join(
            change_schema.transformer_parameter_event.filter(c("side") == "t2"),
            left_on="uuid",
            right_on="eq_fk",
            how="left",
        )
        .join(transformer_end, left_on="uuid", right_on="eq_fk", how="left")
        .join(eq_connectivity, left_on="uuid", right_on="eq_fk", how="left")
        .join(
            node_data.select("node_id", c("v_base").alias("v_base_hv")),
            left_on="u_of_edge",
            right_on="node_id",
            how="left",
        )
        .join(
            node_data.select("node_id", c("v_base").alias("v_base_lv"), "i_base"),
            left_on="v_of_edge",
            right_on="node_id",
            how="left",
        )
        .with_columns(
            ((c("vn_hv") * c("v_base_lv")) / (c("vn_lv") * c("v_base_hv"))).alias(
                "n_transfo"
            ),
            (c("v_base_lv") ** 2 / s_base).alias("z_base"),
        )
        .with_columns(
            c("uuid").alias("eq_fk"),
            (c("rated_s") * 1e3 / (np.sqrt(3) * c("v_base_lv") * c("i_base"))).alias(
                "i_max_pu"
            ),
            (c("r") / c("z_base")).alias("r_pu"),
            (c("x") / c("z_base")).alias("x_pu"),
            pl.lit("transformer").alias("type"),
            (c("rated_s") * 1e3 / s_base).alias("p_max_pu"),
        )
    )

    switch = (
        pl.DataFrame(change_schema.switch)
        .join(eq_connectivity, left_on="uuid", right_on="eq_fk", how="left")
        .join(
            node_data["node_id", "v_base", "i_base"],
            left_on="u_of_edge",
            right_on="node_id",
            how="left",
        )
        .with_columns(
            c("uuid").alias("eq_fk"),
            pl.lit("switch").alias("type"),
        )
    )

    grid_data["edge_data"] = pl.concat(
        [branch, transformer, switch], how="diagonal_relaxed"
    ).with_row_index(name="edge_id")

    node_data = validate_data(grid_data["node_data"], NodeData)
    edge_data = validate_data(grid_data["edge_data"], EdgeData)
    load_data = {}
    for key in grid_data["load_data"]:
        load_data[key] = validate_data(grid_data["load_data"][key], LoadData)

    return NodeEdgeModel(
        node_data=node_data,
        edge_data=edge_data,
        load_data=load_data,
    )
