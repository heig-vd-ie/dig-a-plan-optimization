import os
import copy
import polars as pl
import numpy as np
import pandas as pd
import pandapower as pp
from helpers.json import save_obj_to_json
from konfig import settings, PROJECT_ROOT
from data_model.benchmark import PowerFlowResponse


def check_line_loading(net: pp.pandapowerNet) -> pd.DataFrame:
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

    return df


def check_trafo_loading(net: pp.pandapowerNet) -> pd.DataFrame:
    """
    Returns a transformer table sorted by pandapower res_trafo.loading_percent.
    """

    res_cols = [c for c in ["loading_percent"] if c in net.res_trafo.columns]

    df = net.trafo.copy()
    df = df.join(net.res_trafo[res_cols], how="left")
    df["trafo_idx"] = df.index

    df = df[df["loading_percent"].notna()]
    df = df.sort_values("loading_percent", ascending=False)

    return df


def check_voltage_limits(net: pp.pandapowerNet) -> pd.DataFrame:
    """
    Returns a bus voltage table sorted by pandapower res_bus.vm_pu.
    """

    res_cols = [c for c in ["vm_pu"] if c in net.res_bus.columns]

    df = net.bus.copy()
    df = df.join(net.res_bus[res_cols], how="left")
    df["bus_idx"] = df.index

    df = df[df["vm_pu"].notna()]
    df = df.sort_values("vm_pu", ascending=False)

    return df


def apply_profile_scenario_to_pandapower(
    net0: pp.pandapowerNet,
    load_df: pl.DataFrame,
    pv_df: pl.DataFrame,
    tcol: str,
    cosphi: float,
) -> pp.pandapowerNet:

    net = copy.deepcopy(net0)

    # net.switch["closed"] = True
    net.load[["p_mw", "q_mvar", "sn_mva"]] = 0.0
    net.sgen[["p_mw", "q_mvar", "sn_mva"]] = 0.0

    load_t = (
        load_df.select(["egid", "index", tcol])
        .filter(pl.col("index").is_in(net.load.index.to_list()))
        .rename({tcol: "p_load_kw"})
        .rename({"index": "load_idx"})
        .to_pandas()
    )

    pv_t = (
        pv_df.select(["egid", "index", tcol])
        .rename({tcol: "p_pv_kw"})
        .filter(pl.col("index").is_in(net.sgen.index.to_list()))
        .rename({"index": "load_idx"})
        .to_pandas()
    )

    load_t = load_t.groupby("load_idx", as_index=False)[["p_load_kw"]].sum()
    load_t["p_load_mw"] = load_t["p_load_kw"] / 1000.0
    load_t["q_load_mvar"] = load_t["p_load_mw"] * np.tan(np.arccos(cosphi))

    pv_t = pv_t.groupby("load_idx", as_index=False)[["p_pv_kw"]].sum()
    pv_t["p_pv_mw"] = pv_t["p_pv_kw"] / 1000.0

    net.load["p_mw"] = net.load["p_mw"].astype("float64")
    net.load["q_mvar"] = net.load["q_mvar"].astype("float64")
    net.sgen["p_mw"] = net.sgen["p_mw"].astype("float64")
    net.sgen["q_mvar"] = net.sgen["q_mvar"].astype("float64")

    lt = load_t.copy()
    lt["load_idx"] = lt["load_idx"].astype(int)
    lt = lt.set_index("load_idx")
    net.load.loc[lt.index, "p_mw"] = lt["p_load_mw"].astype(float)
    net.load.loc[lt.index, "q_mvar"] = lt["q_load_mvar"].astype(float)

    pt = pv_t.copy()
    pt["load_idx"] = pt["load_idx"].astype(int)
    pt = pt.set_index("load_idx")
    net.sgen.loc[pt.index, "p_mw"] = pt["p_pv_mw"].astype(float).values
    net.sgen.loc[pt.index, "q_mvar"] = 0
    return net


def heavy_task_powerflow(
    net0: pp.pandapowerNet,
    kace_name: str,
    load_df: pl.DataFrame,
    pv_df: pl.DataFrame,
    t: str,
    cosφ: float,
    year: int,
    scenario: str,
    threshold_current: float,
    threshold_voltage: float,
):
    """Run power flow as one node"""
    net_case = apply_profile_scenario_to_pandapower(
        net0=net0,
        load_df=load_df,
        pv_df=pv_df,
        tcol=t,
        cosphi=cosφ,
    )
    pp.runpp(net_case)

    cong_lines = check_line_loading(net_case)
    cong_trafos = check_trafo_loading(net_case)
    bus_ou = check_voltage_limits(net_case)
    cache_folder = settings.cache.outputs_benchmark

    if not (PROJECT_ROOT / cache_folder / kace_name).exists():
        os.makedirs(str(PROJECT_ROOT / cache_folder / kace_name), exist_ok=True)
    save_obj_to_json(
        obj=cong_lines[["loading_percent"]].to_dict(),
        path_filename=PROJECT_ROOT
        / cache_folder
        / kace_name
        / f"congested_lines_{scenario}_{year}_{t}.json",
    )
    save_obj_to_json(
        obj=cong_trafos[["loading_percent"]].to_dict(),
        path_filename=PROJECT_ROOT
        / cache_folder
        / kace_name
        / f"congested_trafos_{scenario}_{year}_{t}.json",
    )
    save_obj_to_json(
        obj=bus_ou[["vm_pu"]].to_dict(),
        path_filename=PROJECT_ROOT
        / cache_folder
        / kace_name
        / f"ou_buses_{scenario}_{year}_{t}.json",
    )
    bus_ou_out = bus_ou[
        (bus_ou["vm_pu"] >= 1 + threshold_voltage / 100)
        | (bus_ou["vm_pu"] <= 1 - threshold_voltage / 100)
    ][["bus_idx", "vm_pu"]].to_dict(orient="records")
    cong_lines_out = cong_lines[cong_lines["loading_percent"] >= threshold_current][
        ["line_idx", "loading_percent"]
    ].to_dict(orient="records")
    cong_trafos_out = cong_trafos[cong_trafos["loading_percent"] >= threshold_current][
        ["trafo_idx", "loading_percent"]
    ].to_dict(orient="records")
    return PowerFlowResponse(
        congested_lines=cong_lines_out,
        congested_trafos=cong_trafos_out,
        congested_buses=bus_ou_out,
    )
