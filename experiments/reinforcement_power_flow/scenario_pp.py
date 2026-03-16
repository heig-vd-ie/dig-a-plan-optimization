import numpy as np
import pandas as pd
import pandapower as pp
import polars as pl
import copy



def apply_profile_scenario_to_pandapower(
    net0: pp.pandapowerNet,
    profile_df: pl.DataFrame,
    tcol: str,
    mapping_load: pd.DataFrame,
    cosphi: float,
) -> pp.pandapowerNet:
    
    net = copy.deepcopy(net0)

    net.switch["closed"] = True
    net.load.loc[:, ["p_mw", "q_mvar"]] = 0.0
    net.sgen.loc[:, ["p_mw", "q_mvar"]] = 0.0

    load_t = (
        profile_df
        .select(["egid", tcol])
        .rename({tcol: "p_kw"})
        .to_pandas()
    )
    
    load_t["egid"] = load_t["egid"].astype(int)
    load_t["p_kw"] = load_t["p_kw"].astype(float)

    scen = (
        load_t
        .merge(mapping_load[["egid", "load_idx"]], on="egid", how="inner")
        .groupby("load_idx", as_index=False)["p_kw"]
        .sum()
    )

    scen["p_mw"] = scen["p_kw"] / 1000.0
    scen["q_mvar"] = scen["p_mw"] * np.tan(np.arccos(cosphi))

    net.load.loc[scen["load_idx"], "p_mw"] = scen["p_mw"].to_numpy()
    net.load.loc[scen["load_idx"], "q_mvar"] = scen["q_mvar"].to_numpy()

    return net