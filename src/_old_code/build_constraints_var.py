import numpy as np
import cvxpy as cp

def build_problem_constraints(net, T, n_bus, RELAX, p_load, q_load,
                              gen_Plo, gen_Pup, gen_Qlo, gen_Qup, vmin, vmax,
                              p, q, 
                              # Variables for the node-based (downstream) flow:
                              P_dn, Q_dn,      
                              # Variables for the branch (upstream) flow :
                              P_up, Q_up,  
                              f_var, v_sq,
                              upstream, downstream, z, b, v0,
                              genbus, p_gen, q_gen, amp_lim, bus_to_lines,
                              tap_ratios, I_lim_pu, slack_var,
                              # For the external grid:
                              extbus=None,
                              p_ext=None, q_ext=None,
                              ext_Plo=None, ext_Pup=None,
                              ext_Qlo=None, ext_Qup=None):
    """
    Builds constraints for a radial DistFlow model in which the line shunt is split equally.
    
    The power‐flow equations are:
    
        S^Z_dn = - S_node - v * Y_dn - ∑_{child ∈ C} (S^Z_up_child + v * Y_child)
        S^Z_up = i * Z - S^Z_dn    with    i = |S^Z_dn|² / v
    
    and the voltage drop is given by:
    
        v_dn = (v_up/τ²) - 2·Re{ Z* · (S^Z_up) } + |Z|² · i
    ---------------------------------------------------------------
    Inputs:
      • net            : Network data (with net.bus, net.ext_grid, etc.)
      • T              : Number of time steps
      • RELAX          : Boolean; if True use SOC relaxations for current constraints.
      • p_load, q_load : Load profiles per bus
      • gen_*          : Generation limits and profiles (arrays)
      • vmin, vmax     : Voltage limits (in p.u.)
      • p, q : Net injection per bus (real and imaginary parts)
      • P_dn, Q_dn: CVXPY variables representing the downstream (node) flow S^Z_dn
      • P_up, Q_up: CVXPY variables representing the upstream (branch) flow S^Z_up
      • f_var  : CVXPY variables related to the squared current (f_var approximates i)
      • upstream, downstream: Lists of bus indices for each branch (edge)
      • z, b           : Line impedance (complex) and line shunt susceptance arrays
      • v0             : Slack bus voltage magnitude (p.u.)
      • bus_to_lines   : A dict mapping a bus index to the list of branch indices leaving that bus
      • tap_ratios     : Tap ratios for branches
      • I_lim_pu       : Ampacity limits in per unit
      • slack_var      : Additional parameters (if any)
      • extbus, p_ext, q_ext, ext_Plo, ext_Pup, ext_Qlo, ext_Qup: External grid parameters
    ---------------------------------------------------------------
    Note: The voltage is handled as its squared value (v_sq) but the shunt term involves the magnitude v = √(v_sq).
    """
    
    constraints = []
    n_lines = len(upstream)
    n_gen = len(genbus)

    # Get real and imaginary parts of impedance
    r = np.real(z)
    x = np.imag(z)

    # (v_sq is assumed to be defined externally as a CVXPY variable with shape [T, n_bus])
    # Fix slack bus voltage:
    root_bus_id = net.ext_grid.bus.iloc[0]
    root_index = np.where(net.bus.index.values == root_bus_id)[0][0]
    constraints.append(v_sq[: , root_index] == v0**2)

    vmin_sq = vmin**2
    vmax_sq = vmax**2

    # Loop over time steps:
    for t in range(T):
        # Voltage limits at every bus:
        constraints += [v_sq[t, :] >= vmin_sq, v_sq[t, :] <= vmax_sq]

        # Loop over each branch (line)
        for l in range(n_lines):
            up_idx = upstream[l]   # upstream bus index for branch l
            dn_idx = downstream[l] # downstream bus index for branch l
            tau = tap_ratios[l]    # tap ratio for branch l

            # Retrieve squared voltage variables for upstream and downstream buses:
            v_up_sq = v_sq[t, up_idx]
            v_dn_sq = v_sq[t, dn_idx]
            
 

            # Get the list of branch indices leaving the downstream node (children branches)
            child_lines = bus_to_lines.get(dn_idx, [])
            # Sum over the upstream flows from the children:
            sum_P_up = cp.sum(P_up[t, child_lines]) if child_lines else 0
            sum_Q_up = cp.sum(Q_up[t, child_lines]) if child_lines else 0
            # Sum of children shunt contributions: each child branch contributes a shunt of (b[child]/2)*v_up
            if child_lines:
                child_shunt_sum = sum([b[child] for child in child_lines]) / 2.0
            else:
                child_shunt_sum = 0

            # ---------------------------------------------------------------------
            # 1. Node power balance at downstream bus:
            #
            #    S^Z_dn = - S_node - v_up*Y_dn - sum_{child in C}(S^Z_up_child + v_dn*Y_child)
            #
            # Here, Y = -j*(b/2) so that:
            #    v_up * Y has no real part and an imaginary part of + (b/2)*v_up.
            # Therefore, split into real and imaginary parts:
            #
            #   Real part:
            #       P_dn = - p[dn] - sum_{child in C} (P_up_child)
            #
            #   Imaginary part:
            #       Q_dn = - q[dn] + (b[l]/2)*v_up - sum_{child in C} (Q_up_child)
            #                  - ( (sum over children of b[child]/2)*v_dn )
            # ---------------------------------------------------------------------
            
            constraints.append(
                P_dn[t, l] ==  - p[t, dn_idx] - sum_P_up
            )
            constraints.append(
                Q_dn[t, l] ==  - q[t, dn_idx]  - sum_Q_up + (child_shunt_sum)*(v_dn_sq) - (b[l]/2)*v_dn_sq
            )

            # ---------------------------------------------------------------------
            # 2. Define the upstream branch flow on branch l:
            #
            #    S^Z_up = i * Z - S^Z_dn,  where i = |S^Z_dn|² / v_dn.
            # ---------------------------------------------------------------------
            constraints.append(
                P_up[t, l] == r[l]*f_var[t, l] - P_dn[t, l] 
            )
            constraints.append(
                Q_up[t, l] == x[l]*f_var[t, l] - Q_dn[t, l] 
            )

            # ---------------------------------------------------------------------
            # 3. Voltage drop along branch l:
            #
            #    v_dn = (v_up/τ) - 2·Re{ Z* · (S^Z_up) } + |Z|²·i,
            # or equivalently in squared terms:
            #
            #    v_dn_sq = (v_up_sq/(τ²)) - 2*(r[l]*P_up + x[l]*Q_up)
            #              + (r[l]² + x[l]²)*f_var[t, l]
            # ---------------------------------------------------------------------
            expr = (v_up_sq/(tau**2)
                    - 2.0*(r[l]*P_up[t, l] + x[l]*(Q_up[t, l]))
                    + (r[l]**2 + x[l]**2)*f_var[t, l])
            constraints.append(v_dn_sq == expr)

            # ---------------------------------------------------------------------
            # 4. Current (SOC or exact) constraint:
            #
            #    i = |S^Z_up|² / v_up, so we enforce either an SOC constraint or
            #    an exact quadratic equality.
            # ---------------------------------------------------------------------
            if RELAX:
                constraints.append(
                    cp.SOC(
                        (v_up_sq + f_var[t, l]),
                        cp.hstack([
                            2* P_up[t, l], 2* (Q_up[t, l]), f_var[t, l] - v_up_sq   
                        ])
                    )
                )
                               
            else:
                constraints.append(
                    f_var[t, l] * v_up_sq == cp.square(P_up[t, l]) + cp.square(Q_up[t, l])
                )
                

            # ---------------------------------------------------------------------
            # 5. Current limit on physical lines (if applicable):
            # ---------------------------------------------------------------------
            #if amp_lim and l < net.line.shape[0]:
                #constraints.append(f_var[t, l] <= cp.square(I_lim_pu[l]))
                #constraints.append(fbar[t, l] >= f_var[t, l])

    # ---------------------------------------------------------------------
    # 6. Generator constraints (for buses with generators)
    # ---------------------------------------------------------------------
    for t in range(T):
        for g, bus_idx in enumerate(genbus):
            bus_id = np.where(net.bus.index.values == bus_idx)[0][0]
            constraints += [
                p_gen[t, g] >= gen_Plo[g],
                p_gen[t, g] <= gen_Pup[g],
                q_gen[t, g] >= gen_Qlo[g],
                q_gen[t, g] <= gen_Qup[g],
                # Net injection at bus: generation enters with a negative sign
                p[t, bus_id] == - p_gen[t, g] + p_load[t, bus_id],
                q[t, bus_id] == - q_gen[t, g] + q_load[t, bus_id],
            ]

    # ---------------------------------------------------------------------
    # 7. Load-only bus constraints (for buses without generation or external grid)
    # ---------------------------------------------------------------------
    for t in range(T):
        for i in range(n_bus):
            if (i not in genbus) and (extbus is not None and i not in extbus):
                constraints += [
                    p[t, i] ==  p_load[t, i],
                    q[t, i] ==  q_load[t, i],
                ]

    print("\nConstraints built successfully.")
    return constraints
