import polars as pl
from polars import col as c
from data_schema import NodeEdgeModel
import numpy as np
import pandapower as pp
from polars_function import (
    get_transfo_impedance,
    get_transfo_imaginary_component,
)

import patito as pt

from twindigrid_changes import models as changes_models
from twindigrid_changes.schema import ChangesSchema

from general_function import pl_to_dict, snake_to_camel, duckdb_to_dict


def pandapower_to_dig_a_plan_schema(
    net: pp.pandapowerNet, s_base: float = 1e6
) -> NodeEdgeModel:

    grid_data: dict[str, pl.DataFrame] = {}

    bus = net["bus"]
    bus.index.name = "node_id"

    node_data: pl.DataFrame = pl.from_pandas(net.bus.reset_index())
    load: pl.DataFrame = pl.from_pandas(net.load)
    sgen: pl.DataFrame = pl.from_pandas(net.sgen)

    sgen = sgen.group_by("bus").agg(
        (-c("p_mw").sum() * 1e6 / s_base).alias("p_pv"),
        (-c("q_mvar").sum() * 1e6 / s_base).alias("q_pv"),
    )

    load = load.group_by("bus").agg(
        (c("p_mw").sum() * 1e6 / s_base).alias("p_load"),
        (c("q_mvar").sum() * 1e6 / s_base).alias("q_load"),
    )

    load = load.join(sgen, on="bus", how="full", coalesce=True).select(
        c("bus").alias("node_id"),
        pl.sum_horizontal([c("p_load").fill_null(0.0), c("p_pv").fill_null(0.0)]).alias(
            "p_node_pu"
        ),
        pl.sum_horizontal([c("q_load").fill_null(0.0), c("q_pv").fill_null(0.0)]).alias(
            "q_node_pu"
        ),
    )

    node_data = (
        node_data[["node_id", "vn_kv", "name"]]
        .join(load, on="node_id", how="left")
        .select(
            c("name").alias("cn_fk"),
            c("node_id").cast(pl.Int32),
            (c("vn_kv") * 1e3).alias("v_base"),
            c("p_node_pu").fill_null(0.0),
            c("q_node_pu").fill_null(0.0),
            pl.lit(None).cast(pl.Float64).alias("v_node_sqr_pu"),
        )
        .with_columns(
            pl.lit(s_base).alias("s_base"),
            (s_base / (c("v_base") * np.sqrt(3))).alias("i_base"),
        )
    )

    line: pl.DataFrame = pl.from_pandas(net.line)
    line = (
        line.with_columns(
            c("from_bus").cast(pl.Int32).alias("u_of_edge"),
            c("to_bus").cast(pl.Int32).alias("v_of_edge"),
        )
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
            c("name").alias("eq_fk"),
            (c("r_ohm_per_km") * c("length_km") / c("z_base")).alias("r_pu"),
            (c("x_ohm_per_km") * c("length_km") / c("z_base")).alias("x_pu"),
            (
                c("c_nf_per_km") * c("length_km") * 1e-9 * 2 * np.pi * 50 * c("z_base")
            ).alias("b_pu"),
            (c("max_i_ka") * 1e3 / c("i_base")).alias("i_max_pu"),
            pl.lit("branch").alias("type"),
            c("i_base"),
            (np.sqrt(3) * c("max_i_ka") * 1e3 * c("v_base") / s_base).alias("p_max_pu"),
        )
    )
    trafo: pl.DataFrame = pl.from_pandas(net.trafo)

    trafo = (
        trafo.with_columns(
            c("hv_bus").cast(pl.Int32).alias("u_of_edge"),
            c("lv_bus").cast(pl.Int32).alias("v_of_edge"),
        )
        .join(
            node_data.select("node_id", c("v_base").alias("v_base1")),
            left_on="u_of_edge",
            right_on="node_id",
            how="left",
        )
        .join(
            node_data.select("node_id", c("v_base").alias("v_base2"), "i_base"),
            left_on="v_of_edge",
            right_on="node_id",
            how="left",
        )
        .with_columns(
            (c("v_base2") ** 2 / s_base).alias("z_base"),
            (c("vn_hv_kv") / c("v_base1")).alias("vn_hv_pu"),
            ((c("vn_lv_kv") / c("v_base2"))).alias("vn_lv_pu"),
        )
        .with_columns(
            get_transfo_impedance(
                rated_s=c("sn_mva") * 1e6,
                rated_v=c("vn_lv_kv") * 1e3,
                voltage_ratio=c("vk_percent"),
            ).alias("z"),
            get_transfo_impedance(
                rated_s=c("sn_mva") * 1e6,
                rated_v=c("vn_lv_kv") * 1e3,
                voltage_ratio=c("vkr_percent"),
            ).alias("r"),
        )
        .with_columns(
            get_transfo_imaginary_component(module=c("z"), real=c("r")).alias("x"),
        )
        .select(
            "u_of_edge",
            "v_of_edge",
            c("name").alias("eq_fk"),
            (c("r") / c("z_base")).alias("r_pu"),
            (c("x") / c("z_base")).alias("x_pu"),
            pl.lit("transformer").alias("type"),
            (c("sn_mva") * 1e6 / (np.sqrt(3) * c("v_base2") * c("i_base"))).alias(
                "i_max_pu"
            ),
            ((c("vn_hv_pu") / c("vn_lv_pu"))).alias("n_transfo"),
            c("i_base"),
            (c("sn_mva") * 1e6 / s_base).alias("p_max_pu"),
        )
    )

    switch: pl.DataFrame = pl.from_pandas(net.switch)

    switch = switch.filter(c("closed")).select(
        c("name").alias("eq_fk"),
        c("bus").cast(pl.Int32).alias("u_of_edge"),
        c("element").cast(pl.Int32).alias("v_of_edge"),
        pl.lit("switch").alias("type"),
    )

    grid_data["edge_data"] = pl.concat(
        [line, trafo, switch], how="diagonal_relaxed"
    ).with_row_index(name="edge_id")

    ext_grid: pl.DataFrame = pl.from_pandas(net.ext_grid)
    if ext_grid.height != 1:
        raise ValueError("ext_grid should have only 1 row")
    slack_node_id: int = ext_grid["bus"][0]
    v_slack_node_sqr_pu: float = ext_grid["vm_pu"][0] ** 2

    grid_data["node_data"] = node_data.with_columns(
        pl.when(c("node_id") == slack_node_id)
        .then(pl.lit(v_slack_node_sqr_pu))
        .otherwise(c("v_node_sqr_pu"))
        .alias("v_node_sqr_pu"),
        pl.when(c("node_id") == slack_node_id)
        .then(pl.lit("slack"))
        .otherwise(pl.lit("pq"))
        .alias("type"),
    )

    return NodeEdgeModel(
        node_data=grid_data["node_data"],
        edge_data=grid_data["edge_data"],
    )


def change_schema_to_dig_a_plan_schema(
    change_schema: ChangesSchema, s_base: float = 1e6
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

    return NodeEdgeModel(
        node_data=grid_data["node_data"],
        edge_data=grid_data["edge_data"],
    )


def duckdb_to_changes_schema(file_path: str) -> ChangesSchema:
    data = duckdb_to_dict(file_path=file_path)
    schema_dict = {}
    for table_name, table in data.items():
        pt_model: pt.Model = getattr(changes_models, snake_to_camel(table_name))
        schema_dict[table_name] = pt.Model.DataFrame(table).set_model(pt_model).cast()

    return ChangesSchema().replace(**schema_dict)
