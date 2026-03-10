import numpy as np
import pandas as pd
import pandapower as pp
import polars as pl
import copy

def build_snapshot_from_wide_profile(
    profile_df: pl.DataFrame,
    tcol: str,
    mapping_bus: pd.DataFrame,
    cosphi: float,
    s_base: float,
) -> pl.DataFrame:
    load_t = profile_df.select(["egid", tcol]).to_pandas()
    load_t = load_t.rename(columns={tcol: "p_kw"})
    load_t = load_t.dropna(subset=["egid", "p_kw"])
    load_t["egid"] = load_t["egid"].astype(int)

    snapshot = load_t.merge(
        mapping_bus[["egid", "bus"]],
        on="egid",
        how="inner"
    )

    snapshot = (
        snapshot.groupby("bus", as_index=False)["p_kw"]
        .sum()
        .rename(columns={"bus": "node_id"})
    )

    s_base_mva = float(s_base) / 1e6

    snapshot["p_mw"] = snapshot["p_kw"] / 1000.0
    snapshot["p_cons_pu"] = snapshot["p_mw"] / s_base_mva

    tan_phi = np.tan(np.arccos(cosphi))
    snapshot["q_cons_pu"] = snapshot["p_cons_pu"] * tan_phi

    snapshot["p_prod_pu"] = 0.0
    snapshot["q_prod_pu"] = 0.0

    snapshot = snapshot[[
        "node_id",
        "p_cons_pu",
        "q_cons_pu",
        "p_prod_pu",
        "q_prod_pu",
    ]]

    return pl.from_pandas(snapshot)

def apply_profile_scenario_to_pandapower(
    net0: pp.pandapowerNet,
    scenario_df: pl.DataFrame,
    s_base: float,
) -> pp.pandapowerNet:
    
# this function is the bridge between the snapshot table and the pandapower network tables net.load and net.sgen 
    required_cols = {"node_id", "p_cons_pu", "q_cons_pu", "p_prod_pu", "q_prod_pu"}
    missing = required_cols - set(scenario_df.columns)
    if missing:
        raise ValueError(f"Scenario is missing columns: {missing}")

    net = copy.deepcopy(net0)

    # Treat as no-switch network
    if len(net.switch) > 0 and "closed" in net.switch.columns:
        net.switch["closed"] = True

    s_base_mva = float(s_base) / 1e6  

    # Ensure columns exist
    if "q_mvar" not in net.load.columns:
        net.load["q_mvar"] = 0.0
    if "q_mvar" not in net.sgen.columns:
        net.sgen["q_mvar"] = 0.0

    # Reset existing loads & sgens
    net.load["p_mw"] = 0.0
    net.load["q_mvar"] = 0.0
    if len(net.sgen) > 0:
        net.sgen["p_mw"] = 0.0
        net.sgen["q_mvar"] = 0.0

    # Map bus -> load indices / sgen indices
    bus_to_loads = {}
    for lid, bus in net.load["bus"].items():
        bus_to_loads.setdefault(int(bus), []).append(int(lid))

    bus_to_sgens = {}
    for gid, bus in net.sgen["bus"].items():
        bus_to_sgens.setdefault(int(bus), []).append(int(gid))

    # Convert pu -> MW/MVAr
    scen = scenario_df.with_columns(
        (pl.col("p_cons_pu") * s_base_mva).alias("p_load_mw"),
        (pl.col("q_cons_pu") * s_base_mva).alias("q_load_mvar"),
        (pl.col("p_prod_pu") * s_base_mva).alias("p_gen_mw"),
        (pl.col("q_prod_pu") * s_base_mva).alias("q_gen_mvar"),
    )

    # Apply bus by bus
    for row in scen.select(
        ["node_id", "p_load_mw", "q_load_mvar", "p_gen_mw", "q_gen_mvar"]
    ).iter_rows(named=True):
        bus = int(row["node_id"])

        pL = float(row["p_load_mw"])
        qL = float(row["q_load_mvar"])
        pG = float(row["p_gen_mw"])
        qG = float(row["q_gen_mvar"])

        # --- loads ---
        load_ids = bus_to_loads.get(bus, [])
        if len(load_ids) == 0:
            new_id = pp.create_load(
                net, bus=bus, p_mw=0.0, q_mvar=0.0, name=f"auto_load_{bus}"
            )
            load_ids = [int(new_id)]
            bus_to_loads[bus] = load_ids

        share = 1.0 / len(load_ids)
        for lid in load_ids:
            net.load.at[lid, "p_mw"] += pL * share
            net.load.at[lid, "q_mvar"] += qL * share

        # --- generators (PV) ---
        if abs(pG) > 1e-12 or abs(qG) > 1e-12:
            gen_ids = bus_to_sgens.get(bus, [])
            if len(gen_ids) == 0:
                new_gid = pp.create_sgen(
                    net, bus=bus, p_mw=0.0, q_mvar=0.0, name=f"auto_pv_{bus}"
                )
                gen_ids = [int(new_gid)]
                bus_to_sgens[bus] = gen_ids

            gshare = 1.0 / len(gen_ids)
            for gid in gen_ids:
                net.sgen.at[gid, "p_mw"] += pG * gshare
                net.sgen.at[gid, "q_mvar"] += qG * gshare

    return net