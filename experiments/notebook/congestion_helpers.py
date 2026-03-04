from __future__ import annotations

import pandas as pd
import pandapower as pp


def check_congested_lines(
    net: pp.pandapowerNet,
    limit_percent: float = 100.0,
) -> pd.DataFrame:
    """
    Return lines whose loading_percent exceeds the given limit.
    """
    if len(net.line) == 0 or len(net.res_line) == 0:
        return pd.DataFrame()

    df = net.line.copy()
    df = df.join(net.res_line, how="left", rsuffix="_res")
    df["line_idx"] = df.index
    df = df[df["loading_percent"] > limit_percent].copy()

    return df.sort_values("loading_percent", ascending=False)


def check_congested_trafos(
    net: pp.pandapowerNet,
    limit_percent: float = 100.0,
) -> pd.DataFrame:
    """
    Return transformers whose loading_percent exceeds the given limit.
    """
    if len(net.trafo) == 0 or len(net.res_trafo) == 0:
        return pd.DataFrame()

    df = net.trafo.copy()
    df = df.join(net.res_trafo, how="left", rsuffix="_res")
    df["trafo_idx"] = df.index
    df = df[df["loading_percent"] > limit_percent].copy()

    return df.sort_values("loading_percent", ascending=False)


def check_voltage_violations(
    net: pp.pandapowerNet,
    vmin_pu: float = 0.95,
    vmax_pu: float = 1.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return undervoltage and overvoltage bus tables.
    """
    if len(net.res_bus) == 0:
        return pd.DataFrame(), pd.DataFrame()

    bus_df = net.bus.copy()
    bus_df = bus_df.join(net.res_bus, how="left", rsuffix="_res")
    bus_df["bus_idx"] = bus_df.index

    undervoltage = (
        bus_df[bus_df["vm_pu"] < vmin_pu]
        .copy()
        .sort_values("vm_pu", ascending=True)
    )
    overvoltage = (
        bus_df[bus_df["vm_pu"] > vmax_pu]
        .copy()
        .sort_values("vm_pu", ascending=False)
    )

    return undervoltage, overvoltage


def get_line_current_margins(net: pp.pandapowerNet) -> pd.DataFrame:
    """
    Return line current utilization table sorted from highest to lowest.
    """
    if len(net.line) == 0 or len(net.res_line) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(index=net.line.index)
    df["from_bus"] = net.line["from_bus"]
    df["to_bus"] = net.line["to_bus"]
    df["name"] = net.line["name"] if "name" in net.line.columns else ""
    df["max_i_ka"] = net.line["max_i_ka"]
    df["i_from_ka"] = net.res_line["i_from_ka"]
    df["i_to_ka"] = net.res_line["i_to_ka"]
    df["i_max_ka"] = df[["i_from_ka", "i_to_ka"]].max(axis=1)
    df["utilization_percent"] = 100.0 * df["i_max_ka"] / df["max_i_ka"]

    return df.sort_values("utilization_percent", ascending=False)


def reinforce_line_one_step(
    net: pp.pandapowerNet,
    line_idx: int,
    step_percent: float = 20.0,
) -> None:
    """
    Increase line thermal capacity by increasing max_i_ka.
    """
    if "max_i_ka" in net.line.columns and pd.notna(net.line.at[line_idx, "max_i_ka"]):
        net.line.at[line_idx, "max_i_ka"] *= (1.0 + step_percent / 100.0)


def reinforce_trafo_one_step(
    net: pp.pandapowerNet,
    trafo_idx: int,
    step_percent: float = 20.0,
) -> None:
    """
    Increase transformer capacity by increasing sn_mva.
    """
    if (
        len(net.trafo) > 0
        and "sn_mva" in net.trafo.columns
        and pd.notna(net.trafo.at[trafo_idx, "sn_mva"])
    ):
        net.trafo.at[trafo_idx, "sn_mva"] *= (1.0 + step_percent / 100.0)