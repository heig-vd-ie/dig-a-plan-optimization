import numpy as np
from collections import defaultdict
from typing import Any
import polars as pl

def build_input_parameters(net, schema, s_base: float, radial: bool = True) -> dict[str, Any]:
    """
    Extracts and converts all required input parameters from the pandapower net
    and the DistFlow schema. The schema is assumed to be already processed (i.e.
    nodes re-indexed and edges directed) via the data_connector.
    
    All power values are in Watts, voltages in V, etc.
    """
    # 1) Extract node data from the schema.
    node_df = schema.node_data.to_pandas()
    # For the edge data, we assume the schema has been processed already.
    edge_df = schema.edge_data.to_pandas()
    
    # 2) Build node-based arrays.
    bus_ids = node_df["node_id"].values
    n_bus = len(bus_ids)
    
    p_load_array = node_df["p_node_pu"].fillna(0).astype(np.float64).values
    q_load_array = node_df["q_node_pu"].fillna(0).astype(np.float64).values
    
    # 3) Convert edge-based columns to arrays.
    upstream   = edge_df["u_of_edge"].astype(int).values
    downstream = edge_df["v_of_edge"].astype(int).values
    r_pu       = edge_df["r_pu"].astype(np.float64).values
    x_pu       = edge_df["x_pu"].astype(np.float64).values
    b_pu       = edge_df["b_pu"].astype(np.float64).values
    # The transformer tap ratio is stored in the column "n_transfo" for transformers,
    # and for non-transformers the default is 1.
    n_transfo  = edge_df["n_transfo"].astype(np.float64).values
    tap_ratios = np.where(edge_df["type"].values == "transformer", n_transfo, 1.0)
    
    # Combine the branch resistance and reactance into a complex number.
    z_combined = r_pu + 1j * x_pu
    
    # 4) Build bus-to-lines mapping.
    bus_to_lines = defaultdict(list)
    for idx, row in edge_df.iterrows():
        u = int(row["u_of_edge"])
        bus_to_lines[u].append(idx)
    
    # 5) Build an (optional) adjacency matrix.
    G_adj = np.zeros((n_bus, n_bus), dtype=int)
    for idx, row in edge_df.iterrows():
        u = int(row["u_of_edge"])
        v = int(row["v_of_edge"])
        G_adj[u, v] = 1
    
    # 6) Prepare load arrays (Watts) for T=1.
    T = 1
    p_load = np.tile(p_load_array, (T, 1))
    q_load = np.tile(q_load_array, (T, 1))
    
    # 7) Voltage limits and slack voltage.
    if "min_vm_pu" in net.bus.columns:
        vmin = net.bus["min_vm_pu"].values
    else:
        vmin = np.full(n_bus, 0.96)
    if "max_vm_pu" in net.bus.columns:
        vmax = net.bus["max_vm_pu"].values
    else:
        vmax = np.full(n_bus, 1.05)
    
    # Assume the external grid bus is the slack bus.
    external_bus = net.ext_grid["bus"][0]
    v0_pu = net.ext_grid["vm_pu"].iloc[0]
    
    # 8) Generator limits (convert from MW to W, then normalize by s_base).
    gen_Plo = net.gen["min_p_mw"].values * 1e6 / s_base
    gen_Pup = net.gen["max_p_mw"].values * 1e6 / s_base
    gen_Qlo = net.gen["min_q_mvar"].values * 1e6 / s_base
    gen_Qup = net.gen["max_q_mvar"].values * 1e6 / s_base
    
    # 9) Current limits for lines.
    if net.line.shape[0] > 0:
        V_base_lines = net.bus.loc[net.line['from_bus'], 'vn_kv'].values
        I_base_line  = (s_base * 1e6) / (np.sqrt(3) * V_base_lines * 1e3)
        I_lim_line   = (net.line["max_i_ka"].values * 1e3) / I_base_line
    else:
        I_lim_line = np.array([])
        
    lmax_values = I_lim_line
    # We'll define the line-by-line P/Q min/max arrays.
    n_lines_pp = len(net.line)  # number of actual lines in net.line
    p_line_min = np.zeros(n_lines_pp)
    p_line_max = np.zeros(n_lines_pp)
    q_line_min = np.zeros(n_lines_pp)
    q_line_max = np.zeros(n_lines_pp)
    
    for i in range(n_lines_pp):
        # Simple approximation: assume the line can carry sqrt(3)*I_lim_line in p.u. real power
        # at v=1 p.u.  => P_max = sqrt(3)* I_lim_line[i]
        flow_max = np.sqrt(3) * I_lim_line[i]
        p_line_max[i] = flow_max
        p_line_min[i] = -flow_max
        q_line_max[i] = flow_max
        q_line_min[i] = -flow_max
    
    # 10) External grid limits.
    ext_Plo = net.ext_grid["min_p_mw"].values * 1e6 / s_base
    ext_Pup = net.ext_grid["max_p_mw"].values * 1e6 / s_base
    ext_Qlo = np.full_like(net.ext_grid["min_p_mw"].values, -9999.0)
    ext_Qup = np.full_like(net.ext_grid["max_p_mw"].values, 9999.0)
    
    # 11) Define regulator-related parameters.
    # If the net has voltage regulators, we assume net has an attribute "regbus" containing the bus IDs
    # where regulators are installed. Otherwise, regbus is empty.
    if hasattr(net, "regbus"):
        regbus = net.regbus  
    else:
        regbus = np.array([])  # No regulators.
    # Define regratio: shape (T, len(regbus)). If regulators exist, initialize to ones.
    if regbus.size > 0:
        regratio_values = np.full((T, len(regbus)), 1.0)
    else:
        regratio_values = np.empty((T, 0))
    
    # 11) Package all the input arrays into a dictionary.
    inputs_dict = {
        "n_bus": n_bus,
        "T": T,
        "p_load": p_load,
        "q_load": q_load,
        "vmin": vmin,
        "vmax": vmax,
        "v0": v0_pu,  # absolute slack voltage (V)
        "upstream": upstream,
        "downstream": downstream,
        "z": z_combined,
        "b": b_pu,
        "tap_ratios": tap_ratios,
        "bus_to_lines": bus_to_lines,
        "G": G_adj,
        "I_lim_pu": I_lim_line,
        "gen_Plo": gen_Plo,
        "gen_Pup": gen_Pup,
        "gen_Qlo": gen_Qlo,
        "gen_Qup": gen_Qup,
        "extbus": net.ext_grid["bus"].values,
        "ext_Plo": ext_Plo,
        "ext_Pup": ext_Pup,
        "ext_Qlo": ext_Qlo,
        "ext_Qup": ext_Qup,
        "lmax": lmax_values,
        "regbus": regbus,
        "regratio": regratio_values,
        "p_line_min": p_line_min,
        "p_line_max": p_line_max,
        "q_line_min": q_line_min,
        "q_line_max": q_line_max,
    }
    
    # print("Edge DataFrame used for input parameters:")
    # print(edge_df)
   

    
    return inputs_dict
