# constraints.py
import pyomo.environ as pyo

def master_model_constraints(model: pyo.AbstractModel) -> pyo.AbstractModel:

    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)
    model.orientation = pyo.Constraint(model.L, rule=orientation_rule)
    model.flow_P_lower = pyo.Constraint(model.LC, rule=flow_P_lower_rule)
    model.flow_P_upper = pyo.Constraint(model.LC, rule=flow_P_upper_rule)
    model.flow_Q_lower = pyo.Constraint(model.LC, rule=flow_Q_lower_rule)
    model.flow_Q_upper = pyo.Constraint(model.LC, rule=flow_Q_upper_rule)
    model.power_balance_real = pyo.Constraint(model.N, rule=power_balance_real_rule)
    model.power_balance_reactive = pyo.Constraint(model.N, rule=power_balance_reactive_rule)
    model.radiality = pyo.Constraint(model.N, rule=radiality_rule)
    
    return model

# Objective function: minimize approximate losses weighted by line resistances.
def objective_rule(m):
    return sum(
        m.r[l] * (m.p_flow[l, a, b]**2 + m.q_flow[l, a, b]**2)
        for (l, a, b) in m.LC
    )

# Orientation constraint.
def orientation_rule(m, l):

    if l in m.S:
        return sum(m.d[l_, i, j] for (l_, i, j) in m.LC if l_ == l) == m.delta[l]
    else:
        return sum(m.d[l_, i, j] for (l_, i, j) in m.LC if l_ == l) == 1

# Big-M constraints for real power flows.
def flow_P_lower_rule(m, l, i, j):
    return m.p_flow[l, i, j] >= -m.M * m.d[l, i, j]

def flow_P_upper_rule(m, l, i, j):
    return m.p_flow[l, i, j] <= m.M * m.d[l, i, j]

# Big-M constraints for reactive power flows.
def flow_Q_lower_rule(m, l, i, j):
    return m.q_flow[l, i, j] >= -m.M * m.d[l, i, j]

def flow_Q_upper_rule(m, l, i, j):
    return m.q_flow[l, i, j] <= m.M * m.d[l, i, j]

# Real power balance.
def power_balance_real_rule(m, n):
    if n == m.slack_node:
        return pyo.Constraint.Skip
    p_in = sum(m.p_flow[l, a, b] for (l, a, b) in m.LC if b == n)
    p_out = sum(m.p_flow[l, a, b] for (l, a, b) in m.LC if a == n)
    return p_in - p_out == m.p_node[n]

    # Reactive power balance.
def power_balance_reactive_rule(m, n):
    if n == m.slack_node:
        return pyo.Constraint.Skip
    q_in = sum(m.q_flow[l, a, b] for (l, a, b) in m.LC if b == n)
    q_out = sum(m.q_flow[l, a, b] for (l, a, b) in m.LC if a == n)
    return q_in - q_out == m.q_node[n]

# Radiality constraint: each non-slack bus must have one incoming candidate.
def radiality_rule(m, n):
    if n == m.slack_node:
        return sum(m.d[l, a, b] for (l, a, b) in m.LC if a == n) == 0
    else:
        return sum(m.d[l, a, b] for (l, a, b) in m.LC if a == n) == 1