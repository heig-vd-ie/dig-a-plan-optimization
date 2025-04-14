import pyomo.environ as pyo

def slave_model_candidate_indexed():
    """
    Creates a Pyomo AbstractModel for the detailed slave stage using candidate-indexed variables.
    
    In this formulation:
      - The candidate set LF is defined as all tuples (l, i, j), where for branch l candidate (i,j)
        designates that bus i is the upstream (sending) bus and j is the downstream (receiving) bus.
      - The master decision is provided as a parameter master_d over LF (equal to 1 for the chosen candidate, 0 for nonchosen).
      - A new variable f_c is defined such that:
            f_c(l,i,j) = master_d(l,i,j) * f(l,i,j)
      - The upstream flow equations become:
            P_up(l,i,j) = r[l] * f_c(l,i,j) - P_dn(l,i,j)
            Q_up(l,i,j) = x[l] * f_c(l,i,j) - Q_dn(l,i,j)
      - The voltage drop is defined on the squared voltage variable v_sq:
            Let expr = v_sq[i] - 2*(r[l]*P_up(l,i,j) + x[l]*Q_up(l,i,j)) + (r[l]^2+x[l]^2)*f_c(l,i,j).
            Then we enforce two inequalities:
              (a) v_sq[j] - expr >= - M*(1-master_d(l,i,j)) - s(l,i,j)
              (b) v_sq[j] - expr <=   M*(1-master_d(l,i,j)) + s(l,i,j)
            where s(l,i,j) is a nonnegative slack variable.
      - The rotated cone current constraint is written (in squared form) as:
            (2*P_up(l,i,j))^2 + (2*Q_up(l,i,j))^2 + (v_sq[i] - f(l,i,j))^2 <= (v_sq[i] + f(l,i,j))^2.
      - The objective minimizes total losses approximated by
            sum_{(l,i,j) in LF} r[l]*f_c(l,i,j) + penalty_coef * sum_{(l,i,j) in LF} s(l,i,j).
    
    Instance Data Required:
      • I: set of buses.
      • L: set of branches.
      • F: for each branch l in L, a collection of candidate bus pairs.
      • Children: for each bus j, the set of branch indices (from L) for which j is the upstream node.
      • r, x, b: electrical parameters for branch l.
      • p, q: load at each bus.
      • M, vmin, vmax: Big-M constant and voltage limits.
      • master_d: parameter over LF (keyed by (l,i,j)) with value 1 if candidate is active, 0 otherwise.
    """
    model = pyo.AbstractModel()
    
    # ===== Sets =====
    model.I = pyo.Set()               # Bus indices.
    model.L = pyo.Set()               # Branch indices.
    # F: for each branch l, candidate bus pairs.
    model.F = pyo.Set(model.L, within=model.I * model.I)
    # Candidate connectivity set LF: all tuples (l, i, j) for each (i, j) in F[l].
    model.LF = pyo.Set(
        dimen=3,
        initialize=lambda m: [(l, i, j) for l in m.L for (i, j) in m.F[l]]
    )
    # Children: for each bus j, the set of branch indices for which j is the upstream node.
    model.Children = pyo.Set(model.I)
    
    # ===== Parameters =====
    model.r = pyo.Param(model.L)         # Resistance for branch l.
    model.x = pyo.Param(model.L)         # Reactance for branch l.
    model.b = pyo.Param(model.L)         # Shunt susceptance for branch l.
    model.p = pyo.Param(model.I)         # Real load at bus i.
    model.q = pyo.Param(model.I)         # Reactive load at bus i.
    model.M = pyo.Param(initialize=1e4)    # Big-M constant.
    model.vmin = pyo.Param(initialize=0.9) # Minimum voltage (p.u.)
    model.vmax = pyo.Param(initialize=1.1) # Maximum voltage (p.u.)
    # master_d is defined over LF: 1 if candidate is active, else 0.
    model.master_d = pyo.Param(model.LF)
    
    # ===== Decision Variables =====
    # Voltage in squared form (v_sq) for each bus.
    model.v_sq = pyo.Var(model.I, bounds=(model.vmin**2, model.vmax**2))
    
    # Candidate-indexed branch variables.
    model.P_up = pyo.Var(model.LF, domain=pyo.Reals)
    model.Q_up = pyo.Var(model.LF, domain=pyo.Reals)
    model.P_dn = pyo.Var(model.LF, domain=pyo.Reals)
    model.Q_dn = pyo.Var(model.LF, domain=pyo.Reals)
    model.f = pyo.Var(model.LF, domain=pyo.NonNegativeReals)
    # New variable f_c: f_c = master_d * f.
    model.f_c = pyo.Var(model.LF, domain=pyo.NonNegativeReals)
    # Slack variable for voltage drop constraints.
    model.s = pyo.Var(model.LF, domain=pyo.NonNegativeReals)
    
    # ===== Linking Constraint: f_c = master_d * f =====
    def linking_f_rule(m, l, i, j):
        return m.f_c[l, i, j] == m.master_d[l, i, j] * m.f[l, i, j]
    model.linking_f = pyo.Constraint(model.LF, rule=linking_f_rule)
    
    # ===== Constraints =====
    
    # (1) Slack Bus: fix bus 0's voltage squared to 1.0.
    def slack_voltage_rule(m):
        return m.v_sq[0] == 1.0
    model.slack_voltage = pyo.Constraint(rule=slack_voltage_rule)
    
    # (2) Voltage Limits: enforce v_sq[i] in [vmin^2, vmax^2].
    def voltage_limits_rule(m, i):
        return pyo.inequality(m.vmin**2, m.v_sq[i], m.vmax**2)
    model.voltage_limits = pyo.Constraint(model.I, rule=voltage_limits_rule)
    
    # (3) Node Power Balance (Real) for candidate (l,i,j).
    # For candidate (l, i, j), j is the downstream bus.
    def node_power_balance_real_rule(m, l, i, j):
        children_sum = sum(m.P_up[l2, i2, j2]
                           for (l2, i2, j2) in m.LF if (l2 in m.Children[j] and i2 == j))
        return m.P_dn[l, i, j] == m.master_d[l, i, j] * (- m.p[j] - children_sum)
    model.node_power_balance_real = pyo.Constraint(model.LF, rule=node_power_balance_real_rule)
    
    # (4) Node Power Balance (Reactive) for candidate (l,i,j).
    def node_power_balance_reactive_rule(m, l, i, j):
        children_sum_Q = sum(m.Q_up[l2, i2, j2]
                             for (l2, i2, j2) in m.LF if (l2 in m.Children[j] and i2 == j))
        sum_b_children = sum(m.b[l2] for l2 in m.Children[j]) if j in m.Children else 0
        shunt_term = ((sum_b_children - m.b[l]) / 2) * m.v_sq[j]
        return m.Q_dn[l, i, j] == m.master_d[l, i, j] * (- m.q[j] - children_sum_Q + shunt_term)
    model.node_power_balance_reactive = pyo.Constraint(model.LF, rule=node_power_balance_reactive_rule)
    
    # (5) Upstream Flow Definitions for candidate (l,i,j):
    def upstream_flow_real_rule(m, l, i, j):
        return m.P_up[l, i, j] == m.r[l] * m.f_c[l, i, j] - m.P_dn[l, i, j]
    model.upstream_flow_real = pyo.Constraint(model.LF, rule=upstream_flow_real_rule)
    
    def upstream_flow_reactive_rule(m, l, i, j):
        return m.Q_up[l, i, j] == m.x[l] * m.f_c[l, i, j] - m.Q_dn[l, i, j]
    model.upstream_flow_reactive = pyo.Constraint(model.LF, rule=upstream_flow_reactive_rule)
    
    # (6) Voltage Drop along Branch for candidate (l,i,j).
    # Let expr = v_sq[i] - 2*(r[l]*P_up(l,i,j) + x[l]*Q_up(l,i,j)) + (r[l]^2+x[l]^2)*f_c(l,i,j).
    # We then enforce two separate inequalities:
    def voltage_drop_lower_rule(m, l, i, j):
        expr = m.v_sq[i] - 2*(m.r[l]*m.P_up[l, i, j] + m.x[l]*m.Q_up[l, i, j]) + (m.r[l]**2 + m.x[l]**2)*m.f_c[l, i, j]
        return m.v_sq[j] - expr >= - m.M*(1 - m.master_d[l, i, j]) - m.s[l, i, j]
    model.voltage_drop_lower = pyo.Constraint(model.LF, rule=voltage_drop_lower_rule)
    
    def voltage_drop_upper_rule(m, l, i, j):
        expr = m.v_sq[i] - 2*(m.r[l]*m.P_up[l, i, j] + m.x[l]*m.Q_up[l, i, j]) + (m.r[l]**2 + m.x[l]**2)*m.f_c[l, i, j]
        return m.v_sq[j] - expr <= m.M*(1 - m.master_d[l, i, j]) + m.s[l, i, j]
    model.voltage_drop_upper = pyo.Constraint(model.LF, rule=voltage_drop_upper_rule)
    
    # (7) Rotated Cone (SOC) Current Constraint for candidate (l,i,j):
    # Enforce: ||[2*P_up, 2*Q_up, v_sq[i]-f(l,i,j)]||_2 <= v_sq[i]+f(l,i,j)
    # In squared form: (2*P_up)^2 + (2*Q_up)^2 + (v_sq[i] - f)^2 <= (v_sq[i] + f)^2.
    def current_rotated_cone_rule(m, l, i, j):
        lhs = (2*m.P_up[l, i, j])**2 + (2*m.Q_up[l, i, j])**2 + (m.v_sq[i] - m.f[l, i, j])**2
        rhs = (m.v_sq[i] + m.f[l, i, j])**2
        return lhs <= rhs
    model.current_rotated_cone = pyo.Constraint(model.LF, rule=current_rotated_cone_rule)
    
    # (8) Flow Bounds for candidate (l,i,j):
    def flow_bounds_real_rule(m, l, i, j):
        return pyo.inequality(-m.M, m.P_up[l, i, j], m.M)
    model.flow_bounds_real = pyo.Constraint(model.LF, rule=flow_bounds_real_rule)
    
    def flow_bounds_reactive_rule(m, l, i, j):
        return pyo.inequality(-m.M, m.Q_up[l, i, j], m.M)
    model.flow_bounds_reactive = pyo.Constraint(model.LF, rule=flow_bounds_reactive_rule)
    
    # ===== Objective Function =====
    # Minimize total losses (approximated by r * f_c) plus a heavy penalty on the slack variables.
    penalty_coef = 1e6
    def objective_rule(m):
        loss_term = sum(m.r[l] * m.f_c[l, i, j] for (l, i, j) in m.LF)
        slack_penalty = penalty_coef * sum(m.s[l, i, j] for (l, i, j) in m.LF)
        return loss_term + slack_penalty
    model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)
    
    return model
