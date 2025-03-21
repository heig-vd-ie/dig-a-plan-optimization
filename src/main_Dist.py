import numpy as np
import cvxpy as cp
import pandapower as pp
import networkx as nx
import pandas as pd
import copy
import matplotlib.pyplot as plt
from collections import defaultdict
import os
from data_connector import pandapower_to_distflow, validate_node_index_and_edge_direction 
from distflow_schema.node_data import NodeData
from distflow_schema.edge_data import EdgeData

# Import the helper function from input_parameters.py
from input_parameters import build_input_parameters
from build_constraints_var import build_problem_constraints



# --- Main Script ---
if __name__ == "__main__":
    # 1) Load pandapower network from file
    net_file_path = "./modified_cigre_network_lv.p"
    net = pp.from_pickle(net_file_path)


    # 3) Preprocess transformers: swap if needed and zero out losses
    
    net.trafo['vn_hv_kv'] = 20.0
    net.trafo['vn_lv_kv'] = 0.4
    net.trafo['pfe_kw'] = 0.0
    net.trafo['i0_percent'] = 0.0
    net.trafo['shift_degree'] = 0.0
    if "tap_side" not in net.trafo.columns or net.trafo["tap_side"].isnull().all():
        net.trafo["tap_side"] = "hv"

    # 4) Create generators if none exist
    if net.gen.empty:
        pp.create_gen(net, bus=2, p_mw=0.0, vm_pu=1.0,
                      min_p_mw=0.0, max_p_mw=50.0,
                      min_q_mvar=0.0, max_q_mvar=15.0,
                      name="GenBus2")
        pp.create_gen(net, bus=17, p_mw=0.0, vm_pu=1.0,
                      min_p_mw=0.0, max_p_mw=60.0,
                      min_q_mvar=0.0, max_q_mvar=20.0,
                      name="GenBus17")
        pp.create_gen(net, bus=30, p_mw=0.0, vm_pu=1.0,
                      min_p_mw=0.0, max_p_mw=50.0,
                      min_q_mvar=0.0, max_q_mvar=15.0,
                      name="GenBus30")
    else:
        print("Generators already exist in the network.")

    # 5) Define cost functions
    ext_grid_idx = net.ext_grid.index[0]
    pp.create_poly_cost(net, element=net.gen.index[0], et="gen", cp0_eur=0.0, cp1_eur_per_mw=2.0, cp2_eur_per_mw2=0.01  )
    pp.create_poly_cost(net, element=net.gen.index[1], et="gen", cp0_eur=10.0, cp1_eur_per_mw=1.5, cp2_eur_per_mw2=0.02  )
    pp.create_poly_cost(net, element=net.gen.index[2], et="gen", cp0_eur=0.0, cp1_eur_per_mw=1.0, cp2_eur_per_mw2=0.015)
    pp.create_poly_cost(net, element=ext_grid_idx, et="ext_grid", cp0_eur=0.0, cp1_eur_per_mw=2.0 , cp2_eur_per_mw2=0.02)
    net.ext_grid["min_p_mw"] = [0]
    net.ext_grid["max_p_mw"] = [999]
    

    # 6) Convert the pandapower network into the DistFlow schema
    s_base = net.sn_mva * 1e6  # convert MVA to VA
    schema = pandapower_to_distflow(net, s_base=s_base)
    
    schema, nx_tree_grid, node_idx_mapping = validate_node_index_and_edge_direction(schema)
        


    # 7) Build input parameters from the processed schema
    params = build_input_parameters(net, schema, s_base)
    


    # 8) Define optimization decision variables (CVXPY)
    T = params["T"]
    n_bus = params["n_bus"]
    n_gen = net.gen.shape[0]
    n_ext = len(params["extbus"])
    v_sq    = cp.Variable((T, n_bus), nonneg=True)
    vbar    = cp.Variable((T, n_bus), nonneg=True)
    f_var   = cp.Variable((T, len(params["upstream"])), nonneg=True)
    P_dn = cp.Variable((T, len(params["upstream"])))
    Q_dn = cp.Variable((T, len(params["upstream"])))
    P_up = cp.Variable((T, len(params["upstream"])))
    Q_up = cp.Variable((T, len(params["upstream"])))
    p_gen   = cp.Variable((T, n_gen))
    q_gen   = cp.Variable((T, n_gen))
    p  = cp.Variable((T, n_bus))
    q  = cp.Variable((T, n_bus))
    slack_var = cp.Variable((T, len(params["upstream"])), nonneg=True)
    p_ext = cp.Variable((T, n_ext))
    q_ext = cp.Variable((T, n_ext))
    lmax = cp.Parameter((T, len(params["upstream"]))) 
    


    # 9) Build DistFlow constraints using our constraints builder
    constraints = build_problem_constraints(
        net=net, 
        T=params["T"], n_bus= n_bus, 
        RELAX=True,
        p_load=params["p_load"],
        q_load=params["q_load"],
        gen_Plo=params["gen_Plo"],
        gen_Pup=params["gen_Pup"],
        gen_Qlo=params["gen_Qlo"],
        gen_Qup=params["gen_Qup"],
        vmin=params["vmin"],
        vmax=params["vmax"],
        p=p, q=q,P_dn=P_dn, Q_dn=Q_dn,
        P_up = P_up, Q_up= Q_up, 
        f_var=f_var,v_sq=v_sq,
        upstream=params["upstream"],
        downstream=params["downstream"],
        z=params["z"],
        b=params["b"],
        v0=params["v0"],
        genbus=net.gen["bus"].values,
        p_gen=p_gen, q_gen=q_gen,
        amp_lim=True,
        bus_to_lines=params["bus_to_lines"],
        tap_ratios=params["tap_ratios"],
        I_lim_pu=params["I_lim_pu"],
        slack_var=slack_var,  
        extbus=params["extbus"],
        p_ext=p_ext, q_ext=q_ext,
        ext_Plo=params["ext_Plo"],
        ext_Pup=params["ext_Pup"],
        ext_Qlo=params["ext_Qlo"],
        ext_Qup=params["ext_Qup"]
    )
    
    for l in range(net.line.shape[0]):
        print(f"Line {l}: from_bus={net.line.at[l, 'from_bus']}, "
                f"to_bus={net.line.at[l, 'to_bus']}")


    # 10) Define the objective (using a simple polynomial cost for generation)
    mask_gen = (net.poly_cost.et == "gen")
    c2 = net.poly_cost.loc[mask_gen, "cp2_eur_per_mw2"].values * (s_base**2)
    c1 = net.poly_cost.loc[mask_gen, "cp1_eur_per_mw"].values * s_base
    c0 = net.poly_cost.loc[mask_gen, "cp0_eur"].values
    P_Objective = c2[0] * cp.sum_squares(p_gen) + c1[0] * cp.sum(p_gen) + c0[0]
    
    #######################################################
    beta_Q = 0.01
    Q_Objective = beta_Q * (cp.sum_squares(q_gen))

    ########################################################
    alpha_flow = 0.001  # A small weight to force physically correct flows
    Flow_Penalty = alpha_flow * cp.sum(
                    [f_var[t, l] for t in range(T) for l in range(len(params["upstream"]))]
    )

    Total_Objective = P_Objective + Q_Objective + Flow_Penalty
    obj = cp.Minimize(Total_Objective)
    

    #######################################################
    # 11) Solve the optimization problem
    solver_opts = {"solver": cp.SCS, "verbose": True, "max_iters": 1000000, "eps": 1e-4, "normalize": True,
                  "alpha": 1.5}
    problem = cp.Problem(obj, constraints)
    print("\nSolving the DistFlow OPF...")
    try:
        result = problem.solve(**solver_opts)
        status = problem.status
        print(f"Solver finished. Status: {status}")
    except Exception as e:
        print(f"Solver error: {e}")
        status = "error"
        result = None
        
    ########################################################

    # 12) Post-processing and comparison with pandapower PF
    if status in ["optimal", "optimal_inaccurate"]:
        print(f"Objective Value: {result:.4f}")
        v_solution = np.sqrt(v_sq.value[0, :])
        print("\nDistFlow Voltages (p.u.):")
        for i, vm in enumerate(v_solution):
            print(f"  Bus {i}: {vm:.4f}")
         
         
        calc_currents = []
        opt_currents = []  
        table_rows = []  
        # --- RSOC Constraint Consistency Test ---
        print("\nTesting RSOC constraint consistency:")
        for t in range(T):
            for l in range(len(params["upstream"])):
            # Get the downstream bus index for branch l
                up_idx = params["upstream"][l]
                # Retrieve downstream squared voltage from v_sq
                v_up_sq_val = v_sq.value[t, up_idx]
                # Compute current based on St_real and St_imag:
                s_up_real = P_up.value[t, l]
                s_up_imag = Q_up.value[t, l]
                current_calculated = (s_up_real**2 + s_up_imag**2) / v_up_sq_val
                # Get the optimized f_var value (current variable)
                current_optimized = f_var.value[t, l]
                
                print(f"Branch {l}: P_up={s_up_real:.5f}, Q_up={s_up_imag:.5f}")
                
                # Print or store results
                print(f"Branch {l}, up_idx={up_idx}: computed={current_calculated:.6f}, "
                                f"f_var={current_optimized:.6f}, diff={current_calculated - current_optimized:.6e}")
                difference = abs(current_calculated - current_optimized)
                calc_currents.append(current_calculated)
                opt_currents.append(current_optimized)
                # Print the values and their difference:
                        # Append the row as a dictionary
                table_rows.append({
                        "Time": t,
                        "Branch": l,
                        "Computed Current": current_calculated,
                        "Optimized f_var": current_optimized,
                        "Difference": difference
                })
        # Create a pandas DataFrame
        df = pd.DataFrame(table_rows)   
        # Print the DataFrame as a table
        print(df.to_string(index=False))     
        branch_indices = np.arange(len(params["upstream"]))  
            
        plt.figure(figsize=(10, 6))
        plt.plot(branch_indices, calc_currents, 'bo-', label="calculated current")
        plt.plot(branch_indices, opt_currents, 'rx--', label="Optimized current")
        plt.xlabel("Branch Index")
        plt.ylabel("current (p.u.)")
        plt.legend()
        plt.grid(True)
        plt.xticks(np.arange(n_bus), rotation=90)
        plt.show()



        # Disable original generators and create sgens with DistFlow dispatch
        net.gen["in_service"] = False
        for g in range(n_gen):
            bus_id  = net.gen.at[g, 'bus']
            pg_mw   = p_gen.value[0, g] * s_base / 1e6  # convert to MW
            qg_mvar = q_gen.value[0, g] * s_base / 1e6
            pp.create_sgen(net, bus=bus_id, p_mw=pg_mw, q_mvar=qg_mvar, name=f"custom_sgen_{g}")
    else:
        print(f"Solver did not converge. Status: {status}")

    # 13) Run pandapower PF for side-by-side comparison and plot results
    try:
        pp.runpp(net, max_iteration=400, tolerance_mva=1e-4)
        pandapower_voltages = net.res_bus.vm_pu.values
        print("\nPandapower PF Voltages (p.u.):")
        for i, vm in enumerate(pandapower_voltages):
            print(f"  Bus {i}: {vm:.4f}")
        print("\nCompare DistFlow vs. pandapower side-by-side:")
        for i, (vm_df, vm_pp) in enumerate(zip(v_solution, pandapower_voltages)):
            print(f"  Bus {i}: DistFlow={vm_df:.4f}, pandapower={vm_pp:.4f}")
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(n_bus), v_solution, 'bo-', label="Custom DistFlow OPF")
        plt.plot(np.arange(n_bus), pandapower_voltages, 'rx--', label="pandapower PF")
        plt.xlabel("Bus Index")
        plt.ylabel("Voltage Magnitude (p.u.)")
        plt.title("Comparison: DistFlow vs. pandapower")
        plt.legend()
        plt.grid(True)
        plt.xticks(np.arange(n_bus), rotation=90)
        plt.show()
    except Exception as e:
        print(f"Error running pandapower for comparison: {e}")