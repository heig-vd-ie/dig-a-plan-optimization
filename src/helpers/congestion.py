import copy
import polars as pl
import numpy as np
import pandas as pd
import pandapower as pp


def check_line_loading(net: pp.pandapowerNet, limit_percent: float) -> pd.DataFrame:
    """
    Returns a line table sorted by pandapower res_line.loading_percent.
    """

    res_cols = [
        c
        for c in ["loading_percent", "i_from_ka", "i_to_ka"]
        if c in net.res_line.columns
    ]

    df = net.line.copy()
    df = df.join(net.res_line[res_cols], how="left")
    df["line_idx"] = df.index

    df = df[df["loading_percent"].notna()]
    df = df.sort_values("loading_percent", ascending=False)

    df = df[df["loading_percent"] >= limit_percent].copy()

    return df


def check_trafo_loading(net: pp.pandapowerNet, limit_percent: float) -> pd.DataFrame:
    """
    Returns a transformer table sorted by pandapower res_trafo.loading_percent.
    """

    res_cols = [c for c in ["loading_percent"] if c in net.res_trafo.columns]

    df = net.trafo.copy()
    df = df.join(net.res_trafo[res_cols], how="left")
    df["trafo_idx"] = df.index

    df = df[df["loading_percent"].notna()]
    df = df.sort_values("loading_percent", ascending=False)

    df = df[df["loading_percent"] >= limit_percent].copy()

    return df


def check_voltage_limits(
    net: pp.pandapowerNet, limit_percent: float = 5
) -> pd.DataFrame:
    """
    Returns a bus voltage table sorted by pandapower res_bus.vm_pu.
    """

    res_cols = [c for c in ["vm_pu"] if c in net.res_bus.columns]

    df = net.bus.copy()
    df = df.join(net.res_bus[res_cols], how="left")
    df["bus_idx"] = df.index

    df = df[df["vm_pu"].notna()]
    df = df.sort_values("vm_pu", ascending=False)

    df = df[
        (df["vm_pu"] >= 1 + limit_percent / 100)
        | (df["vm_pu"] <= 1 - limit_percent / 100)
    ].copy()

    return df


def reinforce_line_case(
    net: pp.pandapowerNet,
    line_idx: int,
    loading_percent: float,
    limit_percent: float = 90.0,
    margin: float = 1.05,
    max_step_factor=1.20,
) -> pp.pandapowerNet:
    """
    Increase line thermal capacity by increasing max_i_ka.
    """
    factor = (loading_percent / limit_percent) * margin
    factor = min(factor, max_step_factor)
    net.line.at[line_idx, "max_i_ka"] *= factor
    return net


def reinforce_trafo_case(
    net: pp.pandapowerNet,
    trafo_idx: int,
    loading_percent: float,
    limit_percent: float = 90.0,
    margin: float = 1.05,
    max_step_factor=1.20,
) -> pp.pandapowerNet:
    """
    Increase transformer capacity by increasing sn_mva.
    """
    factor = (loading_percent / limit_percent) * margin
    factor = min(factor, max_step_factor)
    net.trafo.at[trafo_idx, "sn_mva"] *= factor
    return net


def apply_profile_scenario_to_pandapower(
    net0: pp.pandapowerNet,
    load_df: pl.DataFrame,
    pv_df: pl.DataFrame,
    tcol: str,
    cosphi: float,
) -> pp.pandapowerNet:

    net = copy.deepcopy(net0)

    # net.switch["closed"] = True
    net.load.loc[:, ["p_mw", "q_mvar"]] = 0.0
    net.sgen.loc[:, ["p_mw", "q_mvar"]] = 0.0

    load_t = (
        load_df.select(["egid", "index", tcol]).rename({tcol: "p_load_kw"}).to_pandas()
    )

    pv_t = (
        pv_df.select(["egid", "index", tcol])
        .rename({tcol: "p_pv_kw"})
        .filter(pl.col("index").is_in(net.sgen.index.to_list()))
        .to_pandas()
    )

    load_t = load_t.groupby("index", as_index=False)[["p_load_kw"]].sum()
    load_t["p_load_mw"] = load_t["p_load_kw"] / 1000.0
    load_t["q_load_mvar"] = load_t["p_load_mw"] * np.tan(np.arccos(cosphi))

    pv_t = pv_t.groupby("index", as_index=False)[["p_pv_kw"]].sum()
    pv_t["p_pv_mw"] = pv_t["p_pv_kw"] / 1000.0

    net.load["p_mw"] = net.load["p_mw"].astype("float64")
    net.load["q_mvar"] = net.load["q_mvar"].astype("float64")
    net.sgen["p_mw"] = net.sgen["p_mw"].astype("float64")
    net.sgen["q_mvar"] = net.sgen["q_mvar"].astype("float64")

    net.load.loc[load_t["index"], "p_mw"] = load_t["p_load_mw"].astype(float).values
    net.load.loc[load_t["index"], "q_mvar"] = load_t["q_load_mvar"].astype(float).values

    net.sgen.loc[pv_t["index"], "p_mw"] = pv_t["p_pv_mw"].astype(float).values
    net.sgen.loc[pv_t["index"], "q_mvar"] = 0
    return net
