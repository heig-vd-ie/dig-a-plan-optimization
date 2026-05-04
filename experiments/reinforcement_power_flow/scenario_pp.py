import numpy as np
import pandas as pd
import pandapower as pp
import polars as pl
import copy



def apply_profile_scenario_to_pandapower(
    net0: pp.pandapowerNet,
    load_df: pl.DataFrame,
    pv_df: pl.DataFrame,
    tcol: str,
    mapping_load: pd.DataFrame,
    load_to_export_sgen: pd.Series,
    cosphi: float,
) -> pp.pandapowerNet:
    
    net = copy.deepcopy(net0)

    # net.switch["closed"] = True
    net.load.loc[:, ["p_mw", "q_mvar"]] = 0.0
    net.sgen.loc[load_to_export_sgen.values, ["p_mw", "q_mvar"]] = 0.0

    load_t = (
        load_df
        .select(["egid", tcol])
        .rename({tcol: "p_load_kw"})
        .to_pandas()
    )
    
    load_t["egid"] = load_t["egid"].astype(int)
    load_t["p_load_kw"] = load_t["p_load_kw"].astype(float)
    
    pv_t = (
        pv_df
        .select(["egid", tcol])
        .rename({tcol: "p_pv_kw"})
        .to_pandas()
    )

    pv_t["egid"] = pv_t["egid"].astype(int)
    pv_t["p_pv_kw"] = pv_t["p_pv_kw"].astype(float)
    

    scen = (
        load_t
        .merge(pv_t, on="egid", how="left")
        .fillna({"p_pv_kw": 0.0})
        .merge(mapping_load[["egid", "load_idx"]], on="egid", how="inner")
        .groupby("load_idx", as_index=False)[["p_load_kw", "p_pv_kw"]]
        .sum()
    )
    

    scen["p_net_kw"] = scen["p_load_kw"] - scen["p_pv_kw"]
    
    scen["p_load_mw"] = np.maximum(scen["p_net_kw"], 0.0) / 1000.0
    scen["p_export_mw"] = np.maximum(-scen["p_net_kw"], 0.0) / 1000.0

    tanphi = np.tan(np.arccos(cosphi))
    scen["q_load_mvar"] = scen["p_load_mw"] * tanphi

    net.load.loc[scen["load_idx"], "p_mw"] = scen["p_load_mw"].to_numpy()
    net.load.loc[scen["load_idx"], "q_mvar"] = scen["q_load_mvar"].to_numpy()
    
    sgen_idx = scen["load_idx"].map(load_to_export_sgen).astype(int)
    net.sgen.loc[sgen_idx, "p_mw"] = scen["p_export_mw"].to_numpy()
    net.sgen.loc[sgen_idx, "q_mvar"] = 0.0

    return net