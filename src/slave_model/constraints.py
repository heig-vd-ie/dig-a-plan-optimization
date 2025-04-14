import pyomo.environ as pyo


def slave_model_constraints(model: pyo.AbstractModel) -> pyo.AbstractModel:
    # model.linking_f = pyo.Constraint(model.LF, rule=linking_f_rule)
    model.slack_voltage = pyo.Constraint(rule=slack_voltage_rule)
    model.voltage_limits = pyo.Constraint(model.N, rule=voltage_limits_rule)
    model.node_active_power_balance = pyo.Constraint(model.LC, rule=node_active_power_balance_rule)
    model.node_reactive_power_balance = pyo.Constraint(model.LC, rule=node_reactive_power_balance_rule)
    model.active_power_flow = pyo.Constraint(model.LC, rule=active_power_flow_rule)
    model.reactive_power_flow = pyo.Constraint(model.LC, rule=reactive_power_flow_rule)
    model.voltage_drop_lower = pyo.Constraint(model.LC, rule=voltage_drop_lower_rule)
    model.voltage_drop_z_upper = pyo.Constraint(model.LC, rule=voltage_drop_z_upper_rule)
    model.current_rotated_cone = pyo.Constraint(model.LC, rule=current_rotated_cone_rule)
    model.flow_bounds_real = pyo.Constraint(model.LC, rule=current_limit_rule)
    return model
    

# (1) Slack Bus: fix bus 0's voltage squared to 1.0.
def slack_voltage_rule(m):
    return m.v_sq[m.slack_node] == m.slack_v_sq
    # return m.v_sq[m.slack_node] == m.slack_voltage

# (3) Node Power Balance (Real) for candidate (l,i,j).
    # For candidate (l, i, j), j is the downstream bus.  
def node_active_power_balance_rule(m, l, i, j):
    children_sum = sum(
        m.p_z_up[l_2, i_2, j_2]
        for (l_2, i_2, j_2) in m.LC if (i_2 == j) and (j_2 != i)
    )
    return m.p_z_dn[l, i, j] == m.master_d[l, i, j] * (- m.p_node[j] - children_sum)


# (4) Node Power Balance (Reactive) for candidate (l,i,j).
def node_reactive_power_balance_rule(m, l, i, j):
    children_sum = sum(
        m.q_z_up[l_2, i_2, j_2] + m.b[l_2] * m.v_sq[j]
        for (l_2, i_2, j_2) in m.LC if (i_2 == j) and (j_2 != i)
    )

    
    return m.q_z_dn[l, i, j] == m.master_d[l, i, j] * (- m.q_node[j] - children_sum)

# (5) Upstream Flow Definitions for candidate (l,i,j):
def active_power_flow_rule(m, l, i, j):
    return m.p_z_up[l, i, j] == m.master_d[l, i, j] * (m.r[l] * m.i_sq[l, i, j] - m.p_z_dn[l, i, j])

def reactive_power_flow_rule(m, l, i, j):
        return m.q_z_up[l, i, j] == m.master_d[l, i, j] * (m.x[l] * m.i_sq[l, i, j] - m.q_z_dn[l, i, j])
    
# (6) Voltage Drop along Branch for candidate (l,i,j).
# Let expr = v_sq[i] - 2*(r[l]*p_z_up(l,i,j) + x[l]*q_z_up(l,i,j)) + (r[l]^2+x[l]^2)*f_c(l,i,j).
# We then enforce two separate inequalities:
def voltage_drop_lower_rule(m, l, i, j):
    dv = - 2*(m.r[l]*m.p_z_up[l, i, j] + m.x[l]*m.q_z_up[l, i, j]) + (m.r[l]**2 + m.x[l]**2)*m.i_sq[l, i, j]
    return m.v_sq[j] - m.v_sq[i] - dv >= - m.M*(1 - m.master_d[l, i, j])

def voltage_drop_z_upper_rule(m, l, i, j):
    dv = m.v_sq[i] - 2*(m.r[l]*m.p_z_up[l, i, j] + m.x[l]*m.q_z_up[l, i, j]) + (m.r[l]**2 + m.x[l]**2)*m.i_sq[l, i, j]
    return m.v_sq[j] - m.v_sq[i] - dv <= m.M*(1 - m.master_d[l, i, j])
# (7) Rotated Cone (SOC) Current Constraint for candidate (l,i,j):
# Enforce: ||[2*p_z_up, 2*q_z_up, v_sq[i]-f(l,i,j)]||_2 <= v_sq[i]+f(l,i,j)
# In squared form: (2*p_z_up)^2 + (2*q_z_up)^2 + (v_sq[i] - f)^2 <= (v_sq[i] + f)^2.
def current_rotated_cone_rule(m, l, i, j):
    lhs = (2*m.p_z_up[l, i, j])**2 + (2*m.q_z_up[l, i, j])**2 + (m.v_sq[i] - m.i_sq[l, i, j])**2
    rhs = (m.v_sq[i] + m.i_sq[l, i, j])**2
    return lhs <= rhs

# (8) Flow Bounds for candidate (l,i,j):
def current_limit_rule(m, l, i, j):
    return pyo.inequality(-m.i_max[i]**2, m.i_sq[l, i, j], m.i_max[i]**2)
    
# (2) Voltage Limits: enforce v_sq[i] in [vmin^2, vmax^2].
def voltage_limits_rule(m, i):
        return pyo.inequality(m.vmin[i]**2, m.v_sq[i], m.vmax[i]**2)

penalty_coef = 1e6
def objective_rule(m):
    loss_term = sum(m.r[l] * m.i_sq[l, i, j] for (l, i, j) in m.LC)
    slack_penalty = penalty_coef * sum(m.s[l, i, j] for (l, i, j) in m.LC)
    return loss_term + slack_penalty