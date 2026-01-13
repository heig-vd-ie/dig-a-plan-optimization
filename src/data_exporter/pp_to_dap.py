from pathlib import Path
import polars as pl
import patito as pt
from polars import col as c
from shapely import from_geojson
from data_model import NodeEdgeModel, NodeData, EdgeData
import numpy as np
import pandapower as pp
from typing import Tuple
from helpers import (
    get_transfo_impedance,
    get_transfo_imaginary_component,
    pl_to_dict,
)
from data_exporter.uncert_to_scens_rand import generate_random_load_scenarios
from data_model import ShortTermUncertaintyProfile
from data_exporter import validate_data
from data_exporter.uncert_to_scens_prof import (
    ScenarioPipelineProfile,
)


def pp_to_dap(
    net: pp.pandapowerNet, s_base: float = 1e6
) -> Tuple[pt.DataFrame[NodeData], pt.DataFrame[EdgeData], float, pl.DataFrame]:
    """
    Convert a pandapower network to DigAPlan schema.
    This function extracts static node and edge data from the pandapower network
    and validates them against the `data_schema` models.
    It also identifies the slack bus and prepares load data.
    """
    # ------------------------
    bus = net["bus"]
    bus.index.name = "node_id"

    node_data: pl.DataFrame = pl.from_pandas(net.bus.reset_index()).with_columns(
        c("node_id").cast(pl.Int32)
    )
    load: pl.DataFrame = pl.from_pandas(net.load).with_columns(c("bus").cast(pl.Int32))
    sgen: pl.DataFrame = pl.from_pandas(net.sgen).with_columns(c("bus").cast(pl.Int32))

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
        c("p_load").fill_null(0.0).alias("p_cons_pu"),
        c("q_load").fill_null(0.0).alias("q_cons_pu"),
        -c("p_pv").fill_null(0.0).alias("p_prod_pu"),
        -c("q_pv").fill_null(0.0).alias("q_prod_pu"),
    )

    # [["node_id", "vn_kv", "name"]]
    node_data = (
        node_data.select(
            c("node_id"),
            c("name"),
            c("vn_kv"),
            (c("min_vm_pu") if "min_vm_pu" in node_data.columns else pl.lit(0.9)).alias(
                "min_vm_pu"
            ),
            (c("max_vm_pu") if "max_vm_pu" in node_data.columns else pl.lit(1.1)).alias(
                "max_vm_pu"
            ),
        )
        .join(load, on="node_id", how="left")
        .select(
            c("name").alias("cn_fk"),
            c("node_id").cast(pl.Int32),
            (c("vn_kv") * 1e3).alias("v_base"),
            c("min_vm_pu").alias("min_vm_pu"),
            c("max_vm_pu").alias("max_vm_pu"),
            c("p_cons_pu").abs().alias("cons_installed").fill_null(0.001),
            c("p_prod_pu").abs().alias("prod_installed").fill_null(0.001),
            c("p_cons_pu").fill_null(0.001),
            c("q_cons_pu").fill_null(0.001),
            c("p_prod_pu").fill_null(0.001),
            c("q_prod_pu").fill_null(0.001),
        )
        .with_columns(
            pl.lit(s_base).alias("s_base"),
            (s_base / (c("v_base") * np.sqrt(3))).alias("i_base"),
        )
    )

    load_data = node_data.select(
        c("node_id"),
        c("cons_installed").map_elements(
            lambda x: x if x > 0.0 else 1e-6, return_dtype=pl.Float64
        ),
        c("prod_installed").map_elements(
            lambda x: x if x > 0.0 else 1e-6, return_dtype=pl.Float64
        ),
        c("p_cons_pu"),
        c("q_cons_pu"),
        c("p_prod_pu"),
        c("q_prod_pu"),
    )

    node_data = node_data.drop(
        ["p_cons_pu", "q_cons_pu", "p_prod_pu", "q_prod_pu"]
    )  # drop load data from node_data, as it is in load_data

    # ext_grid -> slack identification
    ext_grid: pl.DataFrame = pl.from_pandas(net.ext_grid)
    if ext_grid.height != 1:
        raise ValueError("ext_grid should have only 1 row")
    slack_node_id: int = int(ext_grid["bus"][0])
    v_slack_node_sqr_pu: float = float(ext_grid["vm_pu"][0] ** 2)

    node_data = node_data.with_columns(
        pl.when(c("node_id") == slack_node_id)
        .then(pl.lit("slack"))
        .otherwise(pl.lit("pq"))
        .alias("type"),
    )

    # validate static node data
    node_data_pt = pt.DataFrame(node_data).set_model(NodeData).cast(strict=True)

    node_data = node_data_pt.as_polars()

    # ------------------------
    # --- STATIC EDGE DATA ---
    # ------------------------
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
        .with_columns((c("v_base") ** 2 / s_base).alias("z_base"))
        .select(
            "u_of_edge",
            "v_of_edge",
            (c("name") if "eq_fk" not in line.columns else c("eq_fk")).alias("eq_fk"),
            (c("r_ohm_per_km") * c("length_km") / c("z_base")).alias("r_pu"),
            (c("x_ohm_per_km") * c("length_km") / c("z_base")).alias("x_pu"),
            (
                c("c_nf_per_km") * c("length_km") * 1e-9 * 2 * np.pi * 50 * c("z_base")
            ).alias("b_pu"),
            (c("max_i_ka") * 1e3 / c("i_base")).alias("i_max_pu"),
            pl.lit("branch").alias("type"),
            c("length_km"),
            c("i_base"),
            (np.sqrt(3) * c("max_i_ka") * 1e3 * c("v_base") / s_base).alias("p_max_pu"),
            pl.lit([100]).alias("taps"),
        )
    )

    trafo: pl.DataFrame = pl.from_pandas(net.trafo)
    if "name" in trafo.columns and "eq_fk" not in trafo.columns:
        trafo = trafo.rename({"name": "eq_fk"})

    trafo = trafo.with_columns(
        (
            c("tap_min").fill_null(0)
            if "tap_min" in trafo.columns
            else pl.lit(0).alias("tap_min")
        ),
        (
            c("tap_max").fill_null(0)
            if "tap_max" in trafo.columns
            else pl.lit(0).alias("tap_max")
        ),
        (
            c("tap_step_percent").fill_null(0)
            if "tap_step_percent" in trafo.columns
            else pl.lit(0).alias("tap_step_percent")
        ),
        (
            c("tap_neutral").fill_null(0)
            if "tap_neutral" in trafo.columns
            else pl.lit(0).alias("tap_neutral")
        ),
    )
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
            (c("name") if "eq_fk" not in trafo.columns else c("eq_fk")).alias("eq_fk"),
            (c("r") / c("z_base")).alias("r_pu"),
            (c("x") / c("z_base")).alias("x_pu"),
            pl.lit("transformer").alias("type"),
            (c("sn_mva") * 1e6 / (np.sqrt(3) * c("v_base2") * c("i_base"))).alias(
                "i_max_pu"
            ),
            c("i_base"),
            (c("sn_mva") * 1e6 / s_base).alias("p_max_pu"),
            pl.struct(["tap_min", "tap_max", "tap_step_percent", "tap_neutral"]).alias(
                "tap_info"
            ),
        )
    )

    trafo = trafo.with_columns(
        pl.col("tap_info")
        .map_elements(
            lambda tap: [
                100
                + i * tap["tap_step_percent"] / 100
                - tap["tap_neutral"] * tap["tap_step_percent"] / 100
                for i in range(int(tap["tap_min"]), int(tap["tap_max"]) + 1)
            ],
        )
        .alias("taps")
    ).drop("tap_info")

    switch: pl.DataFrame = pl.from_pandas(net.switch)
    if "name" in switch.columns and "eq_fk" not in switch.columns:
        switch = switch.rename({"name": "eq_fk"})

    switch = switch.select(
        (c("name") if "eq_fk" not in switch.columns else c("eq_fk")).alias("eq_fk"),
        c("bus").cast(pl.Int32).alias("u_of_edge"),
        c("element").cast(pl.Int32).alias("v_of_edge"),
        (~c("closed").cast(pl.Boolean)).alias("normal_open"),
        pl.lit("switch").alias("type"),
        pl.lit([100]).alias("taps"),
    )

    ## Mapping the geo
    coord_mapping_pl: pl.DataFrame = (
        pl.from_dataframe(net.bus)
        .with_columns(
            c("geo")
            .map_elements(
                lambda x: list(from_geojson(x).coords)[0],
                return_dtype=pl.List(pl.Float64),
            )
            .alias("coords"),
        )
        .select(["node_id", "coords"])
    )

    coord_mapping_pl = _handle_missing_coords(
        coord_mapping_pl, switch["u_of_edge", "v_of_edge"]
    )

    coord_mapping = pl_to_dict(coord_mapping_pl["node_id", "coords"])

    node_data = node_data.join(coord_mapping_pl, on="node_id", how="left")

    line = _handle_coords_edge_element(line, coord_mapping)
    trafo = _handle_coords_edge_element(trafo, coord_mapping)
    switch = _handle_coords_edge_element(switch, coord_mapping)
    ###

    edge_data = (
        pl.concat([line, trafo, switch], how="diagonal_relaxed")
        .with_row_index(name="edge_id")
        .with_columns(
            pl.lit(0.0).alias("g_pu"),
            c("b_pu").fill_null(0.0).alias("b_pu"),
        )
    )

    edge_data = edge_data.with_columns(
        pl.struct(["eq_fk", "type", "edge_id"]).map_elements(
            lambda x: x["eq_fk"] if x["eq_fk"] else f"{x["type"]}_{x["edge_id"]}"
        )
    )

    node_data_validated = validate_data(node_data, NodeData)
    edge_data_validated = validate_data(edge_data, EdgeData)

    return node_data_validated, edge_data_validated, v_slack_node_sqr_pu, load_data


def _handle_missing_coords(
    coord_mapping_pl: pl.DataFrame, switch_edges: pl.DataFrame
) -> pl.DataFrame:
    ### Check input
    if not (
        "coords" in coord_mapping_pl.columns and "node_id" in coord_mapping_pl.columns
    ):
        raise ValueError("Wrong schema for coord_mapping_pl")
    if not (
        "u_of_edge" in switch_edges.columns and "v_of_edge" in switch_edges.columns
    ):
        raise ValueError("Wrong schema for switch_edges")
    ### Main funtionality
    edges = (
        switch_edges.join(
            coord_mapping_pl, left_on="u_of_edge", right_on="node_id", how="left"
        )
        .rename({"coords": "u_coords"})
        .join(coord_mapping_pl, left_on="v_of_edge", right_on="node_id", how="left")
        .rename({"coords": "v_coords"})
    )
    inferred_from_u = edges.filter(pl.col("u_coords").is_not_null()).select(
        pl.col("v_of_edge").alias("node_id"), pl.col("u_coords").alias("coords")
    )
    inferred_from_v = edges.filter(pl.col("v_coords").is_not_null()).select(
        pl.col("u_of_edge").alias("node_id"), pl.col("v_coords").alias("coords")
    )
    inferred_coords = pl.concat([inferred_from_u, inferred_from_v]).unique(
        subset=["node_id"]
    )
    output_pl = (
        coord_mapping_pl.join(
            inferred_coords, on="node_id", how="outer", suffix="_inferred"
        )
        .with_columns(
            pl.coalesce(pl.col("coords"), pl.col("coords_inferred")).alias("coords")
        )
        .select("node_id", "coords")
    ).unique(subset=["node_id"])
    ### Check output
    if not ("coords" in output_pl.columns and "node_id" in output_pl.columns):
        raise ValueError("Wrong schema in th output")
    if output_pl.filter(pl.col("coords").is_null()).height > 0:
        raise ValueError("Missing coords in the input")
    return output_pl


def _handle_coords_edge_element(edge_element: pl.DataFrame, coord_mapping: dict):
    return (
        edge_element.with_columns(
            pl.concat_list(
                c("u_of_edge", "v_of_edge").replace_strict(coord_mapping, default=None)
            ).alias("coords"),
        )
        .with_columns(
            c("coords").list.gather_every(n=2).alias("x_coords"),
            c("coords").list.gather_every(n=2, offset=1).alias("y_coords"),
        )
        .with_columns(
            pl.concat_list(c("x_coords").list.get(0), c("x_coords").list.mean()).alias(
                "x_coords"
            ),
            pl.concat_list(c("y_coords").list.get(0), c("y_coords").list.mean()).alias(
                "y_coords"
            ),
        )
    )


def pp_to_dap_w_scenarios(
    net: pp.pandapowerNet,
    egid_id_mapping_file: Path | None = None,
    number_of_random_scenarios: int = 10,
    use_random_scenarios: bool = True,
    p_bounds: Tuple[float, float] | None = None,
    q_bounds: Tuple[float, float] | None = None,
    v_bounds: Tuple[float, float] | None = None,
    s_base: float = 1e6,
    seed: int = 42,
    ksop: ShortTermUncertaintyProfile | None = None,
) -> NodeEdgeModel:
    """
    Convert a pandapower network to DigAPlan schema with random load scenarios.
    This function generates random load scenarios based on the provided node data
    and edge data.
    """
    node_data_validated, edge_data_validated, v_slack_node_sqr_pu, load_data = (
        pp_to_dap(net, s_base=s_base)
    )
    if (
        not use_random_scenarios
        and ksop is not None
        and egid_id_mapping_file is not None
    ):
        scenario_pipeline = ScenarioPipelineProfile()
        rand_scenarios = scenario_pipeline.process(ksop=ksop).map2scens(
            egid_id_mapping_file=egid_id_mapping_file,
            id_node_mapping=net.load,
            cosφ=0.95,
            s_base=s_base,
            seed=seed,
        )
    else:
        rand_scenarios = generate_random_load_scenarios(
            node_data=node_data_validated,
            v_slack_node_sqr_pu=v_slack_node_sqr_pu,
            load_data=load_data,
            number_of_random_scenarios=number_of_random_scenarios,
            p_bounds=p_bounds,
            q_bounds=q_bounds,
            v_bounds=v_bounds,
            seed=seed,
        )

    return NodeEdgeModel(
        node_data=node_data_validated,
        edge_data=edge_data_validated,
        load_data=rand_scenarios,
    )
