import pandapower as pp
import polars as pl
from polars import col as c
from general_function import pl_to_dict
from pipelines.dig_a_plan import DigAPlan


def compare_dig_a_plan_with_pandapower(dig_a_plan: DigAPlan, net: pp.pandapowerNet):
    # ─────────── Apply Switch Status & Run AC PF ───────────
    switch_status = pl_to_dict(
        dig_a_plan.model_manager.extract_switch_status().select("eq_fk", ~c("open"))
    )
    net["switch"]["closed"] = net["switch"]["name"].apply(lambda x: switch_status[x])
    pp.runpp(net)

    # ─────────── Compare Voltages ───────────
    node_voltage = dig_a_plan.model_manager.extract_node_voltage()
    node_voltage_pp = pl.from_pandas(net.res_bus.reset_index())[
        "node_id", "vm_pu", "va_degree"
    ]
    node_voltage = node_voltage.join(
        node_voltage_pp, on="node_id", how="left"
    ).with_columns((c("v_pu") - c("vm_pu")).abs().alias("v_diff"))

    edge_current = dig_a_plan.model_manager.extract_edge_current()
    edge_current_pp = pl.concat(
        [
            pl.from_pandas(net.res_line.reset_index())
            .join(
                pl.from_pandas(net.line.reset_index())["index", "name"],
                on="index",
                how="left",
            )
            .select(
                c("name").alias("eq_fk"),
                (1e3 * (c("i_from_ka") + c("i_to_ka")) / 2).alias("i_pp"),
            ),
            pl.from_pandas(net.res_trafo.reset_index())
            .join(
                pl.from_pandas(net.trafo.reset_index())["index", "name"],
                on="index",
                how="left",
            )
            .select(c("name").alias("eq_fk"), (1e3 * c("i_lv_ka")).alias("i_pp")),
        ]
    )

    edge_current = (
        edge_current.join(edge_current_pp, on="eq_fk", how="left")
        .with_columns((c("i_pp") / c("i_base")))
        .with_columns((c("i_pp") - c("i_pu")).abs().alias("i_diff"))
    )

    print(f"Max voltage difference: {node_voltage["v_diff"].max():.1E} pu")
    print(f"Max current difference: {edge_current["i_diff"].max():.1E} pu")

    return node_voltage, edge_current
