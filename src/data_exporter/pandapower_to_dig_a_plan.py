from typing import List
import polars as pl
import patito as pt
from polars import col as c
from data_schema import NodeEdgeModel
from data_schema import NodeData
import numpy as np
import pandapower as pp
from typing import Tuple
from helper_functions import (
    get_transfo_impedance,
    get_transfo_imaginary_component,
)
from data_schema.edge_data import EdgeData
from pipelines.helpers.scenario_utility import generate_random_load_scenarios
from data_exporter import validate_data
from power_profiles.scenario_factory import ScenarioFactory


def pandapower_to_dig_a_plan_schema(
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
        )
    )

    trafo: pl.DataFrame = pl.from_pandas(net.trafo)
    if "name" in trafo.columns and "eq_fk" not in trafo.columns:
        trafo = trafo.rename({"name": "eq_fk"})
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
        )
    )

    switch: pl.DataFrame = pl.from_pandas(net.switch)
    if "name" in switch.columns and "eq_fk" not in switch.columns:
        switch = switch.rename({"name": "eq_fk"})

    switch = switch.select(
        (c("name") if "eq_fk" not in switch.columns else c("eq_fk")).alias("eq_fk"),
        c("bus").cast(pl.Int32).alias("u_of_edge"),
        c("element").cast(pl.Int32).alias("v_of_edge"),
        (~c("closed").cast(pl.Boolean)).alias("normal_open"),
        pl.lit("switch").alias("type"),
    )

    edge_data = (
        pl.concat([line, trafo, switch], how="diagonal_relaxed")
        .with_row_index(name="edge_id")
        .with_columns(
            pl.lit(0.0).alias("g_pu"),
            c("b_pu").fill_null(0.0).alias("b_pu"),
        )
    )

    node_data_validated = validate_data(node_data, NodeData)
    edge_data_validated = validate_data(edge_data, EdgeData)

    return node_data_validated, edge_data_validated, v_slack_node_sqr_pu, load_data


def pandapower_to_dig_a_plan_schema_with_scenarios(
    net: pp.pandapowerNet,
    number_of_random_scenarios: int = 10,
    use_random_scenarios: bool = True,
    taps: List[int] | None = None,
    p_bounds: Tuple[float, float] | None = None,
    q_bounds: Tuple[float, float] | None = None,
    v_bounds: Tuple[float, float] | None = None,
    s_base: float = 1e6,
    seed: int = 42,
    kace: str = "cigre_mv",
) -> NodeEdgeModel:
    """
    Convert a pandapower network to DigAPlan schema with random load scenarios.
    This function generates random load scenarios based on the provided node data
    and edge data.
    """
    node_data_validated, edge_data_validated, v_slack_node_sqr_pu, load_data = (
        pandapower_to_dig_a_plan_schema(net, s_base=s_base)
    )
    if not use_random_scenarios:
        scenario_factory = ScenarioFactory(kace=kace)
        scenario_factory.initialize().generate_operational_scenarios(
            number_of_random_scenarios=number_of_random_scenarios,
            s_base=s_base,
            seed=seed,
            v_bounds=v_bounds,
        )
        rand_scenarios = scenario_factory.rand_scenarios
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
        taps=taps if taps is not None else list(range(95, 105, 1)),
    )
