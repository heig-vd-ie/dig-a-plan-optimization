import re
import polars as pl
from polars import col as c
from typing import Union
import networkx as nx

import numpy as np
import pandapower as pp
from polars_function import (
    get_transfo_admittance, get_transfo_impedance, get_transfo_conductance, get_transfo_imaginary_component)
from networkx_function import (
    generate_tree_graph_from_edge_data, get_all_edge_data)


from twindigrid_changes.schema import ChangesSchema
from twindigrid_sql.entries.equipment_class import EXTERNAL_NETWORK, TRANSFORMER, SWITCH, BRANCH, EXTERNAL_NETWORK, TRANSFORMER

from general_function import pl_to_dict


def pandapower_to_dig_a_plan_schema(net: pp.pandapowerNet, s_base: float=1e6) -> dict[str, pl.DataFrame]:
    
    
    grid_data: dict[str, pl.DataFrame] = {}
    
    node_data: pl.DataFrame = pl.from_pandas(net.bus)
    load: pl.DataFrame = pl.from_pandas(net.load)
    sgen: pl.DataFrame = pl.from_pandas(net.sgen)

    sgen = sgen.group_by("bus").agg(
        (-c("p_mw").sum()*1e6/s_base).alias("p_pv"),
        (-c("q_mvar").sum()*1e6/s_base).alias("q_pv")
    )

    load = load.group_by("bus").agg(
        (c("p_mw").sum()*1e6/s_base).alias("p_load"),
        (c("q_mvar").sum()*1e6/s_base).alias("q_load")
    )

    load = load.join(sgen, on="bus", how="full", coalesce=True).select(
        c("bus").alias("node_id"),
        pl.sum_horizontal([c("p_load").fill_null(0.), c("p_pv").fill_null(0.)]).alias("p_node_pu"),
        pl.sum_horizontal([c("q_load").fill_null(0.), c("q_pv").fill_null(0.)]).alias("q_node_pu")
    )

    node_data = node_data[["vn_kv", "name"]].with_row_index(name="node_id")\
        .join(load, on="node_id", how="left")\
        .select(
            c("name").alias("cn_fk"),
            c("node_id").cast(pl.Int32),
            (c("vn_kv")*1e3).alias("v_base"),
            c("p_node_pu").fill_null(0.0),
            c("q_node_pu").fill_null(0.0),
            pl.lit(None).cast(pl.Float64).alias("v_node_sqr_pu"),
        ).with_columns(
            pl.lit(s_base).alias("s_base"),
            (s_base /(c("v_base") * np.sqrt(3))).alias("i_base")
        )
    i_base_mapping = pl_to_dict(node_data["node_id", "i_base"])
        
    line: pl.DataFrame = pl.from_pandas(net.line)
    line = line\
        .with_columns(
            c("from_bus").cast(pl.Int32).alias("u_of_edge"),
            c("to_bus").cast(pl.Int32).alias("v_of_edge"),
        ).join(node_data["node_id", "v_base", "i_base"], left_on="u_of_edge", right_on="node_id", how="left")\
        .with_columns(
            (c("v_base")**2 /s_base).alias("z_base"), 
        ).select(
            "u_of_edge", "v_of_edge",
            c("name").alias("eq_fk"),
            (c("r_ohm_per_km")*c("length_km")/c("z_base")).alias("r_pu"),
            (c("x_ohm_per_km")*c("length_km")/c("z_base")).alias("x_pu"),
            (c("c_nf_per_km")*c("length_km")*1e-9*2*np.pi*50*c("z_base")).alias("b_pu"),
            (c("max_i_ka") * 1e3 / c("i_base")).alias("i_max_pu"),
            pl.lit("branch").alias("type"), 
            c("i_base")
        )
    trafo: pl.DataFrame = pl.from_pandas(net.trafo)
    
    trafo = trafo.with_columns(
        c("hv_bus").cast(pl.Int32).alias("u_of_edge"),
        c("lv_bus").cast(pl.Int32).alias("v_of_edge"),
    ).join(node_data.select("node_id", c("v_base").alias("v_base1")), left_on="u_of_edge", right_on="node_id", how="left")\
    .join(node_data.select("node_id", c("v_base").alias("v_base2"), "i_base"), left_on="v_of_edge", right_on="node_id", how="left")\
    .with_columns(
        (c("v_base2")**2 / s_base).alias("z_base"), 
        (c("vn_hv_kv")/c("v_base1")).alias("vn_hv_pu"),
        ((c("vn_lv_kv")/c("v_base2"))).alias("vn_lv_pu"),
    ).with_columns(
        get_transfo_impedance(rated_s=c("sn_mva")*1e6, rated_v=c("vn_lv_kv")*1e3, voltage_ratio=c("vk_percent")).alias("z"),
        get_transfo_impedance(rated_s=c("sn_mva")*1e6, rated_v=c("vn_lv_kv")*1e3, voltage_ratio=c("vkr_percent")).alias("r"),
        get_transfo_admittance(rated_s=c("sn_mva")*1e6, rated_v=c("vn_lv_kv")*1e3, oc_current_ratio=c("i0_percent")).alias("y"),
        get_transfo_conductance(rated_v=c("vn_lv_kv")*1e3, iron_losses=c("pfe_kw")*1e3).alias("g"),
    ).with_columns(
        get_transfo_imaginary_component(module = c("z"), real = c("r")).alias("x"),
        get_transfo_imaginary_component(module = c("y"), real = c("g")).alias("b"),
    ).select(
        "u_of_edge", "v_of_edge", 
        c("name").alias("eq_fk"),
        (c("r")/c("z_base")).alias("r_pu"),
        (c("x")/c("z_base")).alias("x_pu"),
        (c("g")*c("z_base")).alias("g_pu"),
        (c("b")*c("z_base")).alias("b_pu"),
        pl.lit("transformer").alias("type"),
        (c("sn_mva") *1e6 / (np.sqrt(3) * c("v_base2") * c("i_base"))).alias("i_max_pu"),
        ((c("vn_hv_pu")/c("vn_lv_pu"))).alias("n_transfo"),
        c("i_base")
        
    )

    switch: pl.DataFrame = pl.from_pandas(net.switch)

    switch = switch.filter(c("closed"))\
        .select(
            c("name").alias("eq_fk"),
            c("bus").cast(pl.Int32).alias("u_of_edge"),
            c("element").cast(pl.Int32).alias("v_of_edge"),
            pl.lit("switch").alias("type"),
        )

    grid_data["edge_data"] = pl.concat(
        [line, trafo, switch], how="diagonal_relaxed"
    ).with_row_index(name="edge_id")

    ext_grid : pl.DataFrame = pl.from_pandas(net.ext_grid)
    if ext_grid.height != 1:
        raise ValueError("ext_grid should have only 1 row")
    slack_node_id: int = ext_grid["bus"][0]
    v_slack_node_sqr_pu: float = ext_grid["vm_pu"][0]**2
    
    grid_data["node_data"] = node_data.with_columns(
        pl.when(c("node_id") == slack_node_id)
        .then(pl.lit(v_slack_node_sqr_pu))
        .otherwise(c("v_node_sqr_pu")).alias("v_node_sqr_pu"),
        pl.when(c("node_id") == slack_node_id)
        .then(pl.lit("slack"))
        .otherwise(pl.lit("pq")).alias("type"),
    )
    
    return grid_data




    

def schema_to_distflow_schema(
    changes_schema: ChangesSchema, s_base: float
) :
    """
    Convert the schema to a DistFlowSchema instance and add the edge_data table to it
    Also convert in PU the branch parameter

    Args:
        changes_schema (ChangesSchema): The schema to convert from ChangesSchema to DistFlowSchema
        s_base (float): The base power of the system

    Returns:
        dict(distflow_schema:DistFlowSchema,slack_node_id:int): The DistFlowSchema instance and the slack node id

    """

    ##TODO Put s_base in metadata
    ##TODO Add the normalised error, need to be treated and add in schema

    resource_data = changes_schema.resource.filter(
        c("concrete_class").is_in([BRANCH, SWITCH, TRANSFORMER, EXTERNAL_NETWORK])
    ).select(
        c("uuid"), c("dso_code").alias("element_id"), c("concrete_class").alias("type")
    )

    ## Add row index to connectivity_node to give the number of node
    ## Join connectivity_node with connectivity to get the eq_fk
    ## Create the connectivity_node dictionnary with side+eq_fk as key and node index as value
    cn_mapping: dict[float, str] = pl_to_dict(
        changes_schema.connectivity_node.with_row_index()
        .join(changes_schema.connectivity, left_on="uuid", right_on="cn_fk", how="left")
        .select((c("side") + c("eq_fk")).alias("eq_fk_side"), c("index"))
    )

    ## Add node from and node to for each edge with side+eq_fk
    resource_data = resource_data.with_columns(
        ("t1" + c("uuid"))
        .replace_strict(cn_mapping, default=None)
        .alias(
            "u_of_edge"
        ),  # Replace side+eq_fk with node number from connectivity for equipment
        ("t2" + c("uuid")).replace_strict(cn_mapping, default=None).alias("v_of_edge"),
    )

    # Add branch parameter to line_data
    resource_data = resource_data.join(
        changes_schema.branch_parameter_event[["uuid", "eq_fk", "r", "x", "b", "g"]],
        left_on="uuid",
        right_on="eq_fk",
        how="left",
    ).drop("uuid_right")

    ## Add n_tranfo to 1, useless, because automatically added by add_table from class DistFlowSchema and by default value is 1
    # resource_data = resource_data.with_columns(pl.lit(1).alias("n_tranfo"))

    ## Search slack node id for DisFlowSchema
    slack_node_id:int = resource_data.filter(c("type") == EXTERNAL_NETWORK)[
        "u_of_edge"
    ].item()

    ## Remove the external network from the resource_data
    resource_data = resource_data.filter(c("type") != EXTERNAL_NETWORK)

    ## Transfo in pu

    u_b = changes_schema.base_voltage["nominal_voltage"].item()  # V
    i_b = s_base / (3**0.5 * u_b)  # A
    z_b = u_b**2 / s_base  # Ohm
    b_b = 1 / z_b  # S

    pu_base = {
        "U_b": u_b,
        "I_b": i_b,
        "Z_b": z_b,
        "B_b": b_b,
        "S_base": s_base,
    }

    resource_data_pu = resource_data.with_columns(
        (c("g") * pu_base["Z_b"]).alias("g_pu"),
        (c("r") / pu_base["Z_b"]).alias("r_pu"),
        (c("x") / pu_base["Z_b"]).alias("x_pu"),
        (c("b") / pu_base["B_b"]).alias("b_pu"),
    ).drop(["r", "x", "b", "g"])

    ## Check if B is negativ for branch and positiv for trafos
    ## If type is branch, then b_pu is negative, otherwise positive (trafo and switch)
    resource_data_pu = resource_data_pu.with_columns(
        pl.when(c("type") == BRANCH)
        .then(c("b_pu").abs().neg())
        .otherwise(c("b_pu").abs())
        .alias("b_pu")
    )

    ## Create the DistFlowSchema instance