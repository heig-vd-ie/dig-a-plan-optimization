import numpy as np
import pandas as pd
import pandapower as pp
import polars as pl
import copy



def apply_profile_scenario_to_pandapower(
    net0: pp.pandapowerNet,
    load_data: pl.DataFrame,
    pv_data: pl.DataFrame,
    tcol: str,
    mapping_load: pl.DataFrame,
    pv_egid_to_sgen: pl.DataFrame,
    cosphi: float,
) -> pp.pandapowerNet:
    
    net = copy.deepcopy(net0)

    # net.switch["closed"] = True
    net.load.loc[:, ["p_mw", "q_mvar"]] = 0.0
    pv_sgen_idx = pv_egid_to_sgen["sgen_idx"].to_list()
    net.sgen.loc[pv_sgen_idx, ["p_mw", "q_mvar"]] = 0.0

    load_scen = (
        load_data
        .select(["egid", tcol])
        .rename({tcol: "p_load_kw"})
        .with_columns([
            pl.col("egid").cast(pl.Int64, strict=False),
            pl.col("p_load_kw").cast(pl.Float64, strict=False),
        ])
        .join(
            mapping_load.select(["egid", "load_idx"]),
            on="egid",
            how="inner",
        )
        .group_by("load_idx")
        .agg(pl.col("p_load_kw").sum())
        .with_columns([
            (pl.col("p_load_kw") / 1000.0).alias("p_load_mw")
        ])
        .sort("load_idx")
    )
    
    load_scen_pd = load_scen.to_pandas()
    valid_load_idx = load_scen_pd["load_idx"].isin(net.load.index)
    load_scen_pd = load_scen_pd[valid_load_idx].copy()
    tanphi = np.tan(np.arccos(cosphi))
    load_scen_pd["q_load_mvar"] = load_scen_pd["p_load_mw"] * tanphi
    
    net.load.loc[load_scen_pd["load_idx"], "p_mw"] = load_scen_pd["p_load_mw"].values
    net.load.loc[load_scen_pd["load_idx"], "q_mvar"] = load_scen_pd["q_load_mvar"].values

    pv_scen = (
        pv_data
        .select(["egid", tcol])
        .rename({tcol: "p_pv_kw"})
        .with_columns([
            pl.col("egid").cast(pl.Int64, strict=False),
            pl.col("p_pv_kw").cast(pl.Float64, strict=False),
        ])
        .join(pv_egid_to_sgen, on="egid", how="left")
        .drop_nulls(["sgen_idx"])
        .with_columns([
            (pl.col("p_pv_kw") / 1000.0).alias("p_pv_mw"),
            pl.col("sgen_idx").cast(pl.Int64, strict=False),
        ])
        .sort("sgen_idx")
    )

    pv_scen_pd = pv_scen.to_pandas()
    valid_sgen_idx = pv_scen_pd["sgen_idx"].isin(net.sgen.index)
    pv_scen_pd = pv_scen_pd[valid_sgen_idx].copy()

    net.sgen.loc[pv_scen_pd["sgen_idx"], "p_mw"] = pv_scen_pd["p_pv_mw"].values
    net.sgen.loc[pv_scen_pd["sgen_idx"], "q_mvar"] = 0.0
    
    return net