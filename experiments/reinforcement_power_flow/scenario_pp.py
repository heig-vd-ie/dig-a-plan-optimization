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
    pv_egid_to_sgen: pd.Series,
    cosphi: float,
) -> pp.pandapowerNet:
    
    net = copy.deepcopy(net0)

    # net.switch["closed"] = True
    net.load.loc[:, ["p_mw", "q_mvar"]] = 0.0
    net.sgen.loc[pv_egid_to_sgen.values, ["p_mw", "q_mvar"]] = 0.0

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
    

    load_scen = (
        load_t
        .merge(mapping_load[["egid", "load_idx"]], on="egid", how="inner")
        .groupby("load_idx", as_index=False)[["p_load_kw"]]
        .sum()
    )
    load_scen["p_load_mw"] = load_scen["p_load_kw"] / 1000.0
    tanphi = np.tan(np.arccos(cosphi))
    load_scen["q_load_mvar"] = load_scen["p_load_mw"] * tanphi
    
    pv_scen = pv_t.copy()
    pv_scen["sgen_idx"] = pv_scen["egid"].map(pv_egid_to_sgen)
    pv_scen = pv_scen.dropna(subset=["sgen_idx"]).copy()
    pv_scen["sgen_idx"] = pv_scen["sgen_idx"].astype(int)
    pv_scen["p_pv_mw"] = pv_scen["p_pv_kw"] / 1000.0

    net.load.loc[load_scen["load_idx"], "p_mw"] = load_scen["p_load_mw"].tolist()
    net.load.loc[load_scen["load_idx"], "q_mvar"] = load_scen["q_load_mvar"].tolist()
    
    net.sgen.loc[pv_scen["sgen_idx"], "p_mw"] = pv_scen["p_pv_mw"].tolist()
    net.sgen.loc[pv_scen["sgen_idx"], "q_mvar"] = 0.0
    
    return net