from __future__ import annotations

import pandas as pd
import pandapower as pp


def check_line_loading(
    net: pp.pandapowerNet,
    limit_percent: float | None = None,
) -> pd.DataFrame:
    """
    Returns a line table sorted by pandapower res_line.loading_percent.

    - If limit_percent is None: returns ALL lines ranked (most loaded first).
    - If limit_percent is a number (e.g., 100): returns only lines with
      loading_percent > limit_percent (congested lines).

    Note: pandapower loading_percent accounts for max_i_ka, df, and parallel.
    """
    if len(net.line) == 0 or len(net.res_line) == 0:
        return pd.DataFrame()

    res_cols = [c for c in ["loading_percent", "i_from_ka", "i_to_ka"] if c in net.res_line.columns]

    df = net.line.copy()
    df = df.join(net.res_line[res_cols], how="left")
    df["line_idx"] = df.index

    df = df[df["loading_percent"].notna()]
    df = df.sort_values("loading_percent", ascending=False)

    if limit_percent is not None:
        df = df[df["loading_percent"] > limit_percent].copy()

    return df


def check_trafo_loading(
    net: pp.pandapowerNet,
    limit_percent: float | None = None,
) -> pd.DataFrame:
    """
    Returns a transformer table sorted by pandapower res_trafo.loading_percent.

    - If limit_percent is None: returns ALL trafos ranked (most loaded first).
    - If limit_percent is a number (e.g., 100): returns only trafos with
      loading_percent > limit_percent (congested trafos).
    """
    if len(net.trafo) == 0 or len(net.res_trafo) == 0:
        return pd.DataFrame()

    res_cols = [c for c in ["loading_percent"] if c in net.res_trafo.columns]

    df = net.trafo.copy()
    df = df.join(net.res_trafo[res_cols], how="left")
    df["trafo_idx"] = df.index

    df = df[df["loading_percent"].notna()]
    df = df.sort_values("loading_percent", ascending=False)

    if limit_percent is not None:
        df = df[df["loading_percent"] > limit_percent].copy()

    return df


def check_voltage_violations(
    net: pp.pandapowerNet,
    vmin_pu: float = 0.95,
    vmax_pu: float = 1.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return undervoltage and overvoltage bus tables based on res_bus.vm_pu.
    """
    if len(net.res_bus) == 0:
        return pd.DataFrame(), pd.DataFrame()

    bus_df = net.bus.copy()
    bus_df = bus_df.join(net.res_bus, how="left", rsuffix="_res")
    bus_df["bus_idx"] = bus_df.index

    bus_df = bus_df[bus_df["vm_pu"].notna()]

    undervoltage = bus_df[bus_df["vm_pu"] < vmin_pu].copy().sort_values("vm_pu", ascending=True)
    overvoltage  = bus_df[bus_df["vm_pu"] > vmax_pu].copy().sort_values("vm_pu", ascending=False)

    return undervoltage, overvoltage


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
    if "sn_mva" in net.trafo.columns and pd.notna(net.trafo.at[trafo_idx, "sn_mva"]):
        net.trafo.at[trafo_idx, "sn_mva"] *= (1.0 + step_percent / 100.0)