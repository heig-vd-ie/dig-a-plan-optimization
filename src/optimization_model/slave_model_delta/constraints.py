import pyomo.environ as pyo

def slave_model_constraints(model: pyo.AbstractModel) -> pyo.AbstractModel:
    # (1) Objective
    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    # (2) Orientation & Radiality (these will be tautologies once master_d and master_delta are fixed)
    model.orientation       = pyo.Constraint(model.L, rule=orientation_rule)
    model.radiality         = pyo.Constraint(model.N, rule=radiality_rule)

    # (3) Slack‐Voltage: fix slack‐node voltage squared
    model.slack_voltage     = pyo.Constraint(model.N, rule=slack_voltage_rule)
    # (4) Node Real Power Balance
    model.pos_inactive_node_active_power_balance = pyo.Constraint(
        model.LC, rule=pos_inactive_node_active_power_balance_rule
    )
    model.neg_inactive_node_active_power_balance = pyo.Constraint(
        model.LC, rule=neg_inactive_node_active_power_balance_rule
    )
    model.pos_active_node_active_power_balance   = pyo.Constraint(
        model.LC, rule=pos_active_node_active_power_balance_rule
    )
    model.neg_active_node_active_power_balance   = pyo.Constraint(
        model.LC, rule=neg_active_node_active_power_balance_rule
    )

    # (5) Node Reactive Power Balance
    model.pos_inactive_node_reactive_power_balance = pyo.Constraint(
        model.LC, rule=pos_inactive_node_reactive_power_balance_rule
    )
    model.neg_inactive_node_reactive_power_balance = pyo.Constraint(
        model.LC, rule=neg_inactive_node_reactive_power_balance_rule
    )
    model.pos_active_node_reactive_power_balance   = pyo.Constraint(
        model.LC, rule=pos_active_node_reactive_power_balance_rule
    )
    model.neg_active_node_reactive_power_balance   = pyo.Constraint(
        model.LC, rule=neg_active_node_reactive_power_balance_rule
    )

    # (6) Voltage‐Drop Constraints
    model.voltage_drop_lower   = pyo.Constraint(model.LC, rule=voltage_drop_lower_rule)
    model.voltage_drop_upper   = pyo.Constraint(model.LC, rule=voltage_drop_upper_rule)

    # (7) Rotated‐Cone (SOC) Current Constraint
    model.current_rotated_cone = pyo.Constraint(model.LC, rule=current_rotated_cone_rule)

    # (8) Branch Current Limits
    model.current_limit        = pyo.Constraint(model.LC, rule=current_limit_rule)

    # (9) Voltage Limits at each bus
    model.voltage_upper_limits = pyo.Constraint(model.N, rule=voltage_upper_limits_rule)
    model.voltage_lower_limits = pyo.Constraint(model.N, rule=voltage_lower_limits_rule)

    return model


# ----------------------------------------------------------------------------
# 1) Objective: losses + slack penalties
def objective_rule(m):
    edge_losses = sum(m.r[l] * m.i_sq[l, i, j] for (l, i, j) in m.LC)
    v_penalty   = m.penalty_cost * sum(m.slack_v_pos[n] + m.slack_v_neg[n] for n in m.N)
    i_penalty   = m.penalty_cost * sum(m.slack_i_sq[l, i, j] for (l, i, j) in m.LC)
    return edge_losses + v_penalty + i_penalty


# 2) Orientation: ties d to delta (both are fixed by Master already)
def orientation_rule(m, l):
    if l in m.S:
        return sum(m.d[l_, i, j] for (l_, i, j) in m.LC if l_ == l) == m.master_delta[l]
    else:
        return sum(m.d[l_, i, j] for (l_, i, j) in m.LC if l_ == l) == 1


# 3) Radiality: exactly one incoming branch per non‐slack bus
def radiality_rule(m, n):
    if n == pyo.value(m.slack_node):
        return sum(m.d[l, i, j] for (l, i, j) in m.LC if i == n) == 0
    else:
        return sum(m.d[l, i, j] for (l, i, j) in m.LC if i == n) == 1


# 4) Slack Bus Voltage
def slack_voltage_rule(m, n):
    if n == pyo.value(m.slack_node):
        return m.v_sq[n] == m.slack_node_v_sq
    return pyo.Constraint.Skip


# 5) Node Real Power Balance
def pos_inactive_node_active_power_balance_rule(m, l, i, j):
    return m.p_flow[l, i, j] <= m.d[l, i, j] * m.big_m

def neg_inactive_node_active_power_balance_rule(m, l, i, j):
    return m.p_flow[l, i, j] >= -m.d[l, i, j] * m.big_m

def pos_active_node_active_power_balance_rule(m, l, i, j):
    # Sum over child arcs feeding into node i
    downstream_power_flow = sum(
        m.r[l_] * m.i_sq[l_, i_, j_] - m.p_flow[l_, i_, j_]
        for (l_, i_, j_) in m.LC
        if (j_ == i) and (i_ != j)
    )
    return m.p_flow[l, i, j] <= m.p_node[i] + downstream_power_flow + (1 - m.d[l, i, j]) * m.big_m
    

def neg_active_node_active_power_balance_rule(m, l, i, j):
    downstream_power_flow = sum(
        m.r[l_] * m.i_sq[l_, i_, j_] - m.p_flow[l_, i_, j_]
        for (l_, i_, j_) in m.LC
        if (j_ == i) and (i_ != j)
    )
    return m.p_flow[l, i, j] >= m.p_node[i] + downstream_power_flow - (1 - m.d[l, i, j]) * m.big_m
    


# 6) Node Reactive Power Balance
def pos_inactive_node_reactive_power_balance_rule(m, l, i, j):
    return m.q_flow[l, i, j] <= m.d[l, i, j] * m.big_m

def neg_inactive_node_reactive_power_balance_rule(m, l, i, j):
    return m.q_flow[l, i, j] >= -m.d[l, i, j] * m.big_m

def pos_active_node_reactive_power_balance_rule(m, l, i, j):
    downstream_power_flow = sum(
        m.x[l_] * m.i_sq[l_, i_, j_] - m.q_flow[l_, i_, j_]
        for (l_, i_, j_) in m.LC
        if (j_ == i) and (i_ != j)
    )
    transversal_power = sum(
        -m.b[l_] / 2 * m.v_sq[i]
        for (l_, i_, _) in m.LC
        if (i_ == i)
    )
    return m.q_flow[l, i, j] <= m.q_node[i] + downstream_power_flow + transversal_power+ (1 - m.d[l, i, j]) * m.big_m


def neg_active_node_reactive_power_balance_rule(m, l, i, j):
    downstream_power_flow = sum(
        m.x[l_] * m.i_sq[l_, i_, j_] - m.q_flow[l_, i_, j_]
        for (l_, i_, j_) in m.LC
        if (j_ == i) and (i_ != j)
    )
    transversal_power = sum(
        -m.b[l_] / 2 * m.v_sq[i]
        for (l_, i_, _) in m.LC
        if (i_ == i)
    )
    return m.q_flow[l, i, j] >=  m.q_node[i] + downstream_power_flow + transversal_power - (1 - m.d[l, i, j]) * m.big_m
    


# 7) Voltage Drop along each branch (l,i,j)
def voltage_drop_lower_rule(m, l, i, j):
    dv = 2 * (m.r[l] * m.p_flow[l, i, j] + m.x[l] * m.q_flow[l, i, j]) \
         - (m.r[l]**2 + m.x[l]**2) * m.i_sq[l, i, j]
    return (
        m.v_sq[i] / (m.n_transfo[l, i, j]**2)
        - m.v_sq[j] / (m.n_transfo[l, j, i]**2)
        - dv
        >= -m.big_m * (1 - m.d[l, i, j])
    )

def voltage_drop_upper_rule(m, l, i, j):
    dv = 2 * (m.r[l] * m.p_flow[l, i, j] + m.x[l] * m.q_flow[l, i, j]) \
         - (m.r[l]**2 + m.x[l]**2) * m.i_sq[l, i, j]
    return (
        m.v_sq[i] / (m.n_transfo[l, i, j]**2)
        - m.v_sq[j] / (m.n_transfo[l, j, i]**2)
        - dv
        <= m.big_m * (1 - m.d[l, i, j])
    )


# 8) Rotated Cone (SOC) Current Constraint
def current_rotated_cone_rule(m, l, i, j):
    if l in m.S:
        return m.i_sq[l, i, j] == 0
    else:
        lhs = (2 * m.p_flow[l, i, j])**2 \
                + (2 * m.q_flow[l, i, j])**2 \
                + (m.v_sq[i] / (m.n_transfo[l, i, j]**2) - m.i_sq[l, i, j])**2
        rhs = (m.v_sq[i] / (m.n_transfo[l, i, j]**2) + m.i_sq[l, i, j])**2
        return lhs <= rhs


# 9) Branch Current Limits
def current_limit_rule(m, l, i, j):
    return m.i_sq[l, i, j] <= m.i_max[l]**2 + m.slack_i_sq[l, i, j]


# 10) Voltage Limits at each bus n
def voltage_upper_limits_rule(m, n):
    return m.v_sq[n] <= m.v_max[n]**2 + m.slack_v_pos[n]

def voltage_lower_limits_rule(m, n):
    return m.v_sq[n] >= m.v_min[n]**2 - m.slack_v_neg[n]
