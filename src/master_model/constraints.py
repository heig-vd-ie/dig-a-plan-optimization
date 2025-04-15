# constraints.py
import pyomo.environ as pyo

def master_model_constraints(model: pyo.AbstractModel) -> pyo.AbstractModel:
    # Orientation constraint.
    def orientation_rule(m, l):
        if l in m.S:
            return sum(m.d[l_, i, j] for (l_, i, j) in m.LC if l_ == l) == m.Delta[l]
        else:
            return sum(m.d[l_, i, j] for (l_, i, j) in m.LC if l_ == l) == 1
    model.orientation = pyo.Constraint(model.L, rule=orientation_rule)
    
    # Big-M constraints for real power flows.
    def flow_P_lower_rule(m, l, i, j):
        return m.P[l, i, j] >= -m.M * m.d[l, i, j]
    model.flow_P_lower = pyo.Constraint(model.LC, rule=flow_P_lower_rule)
    
    def flow_P_upper_rule(m, l, i, j):
        return m.P[l, i, j] <= m.M * m.d[l, i, j]
    model.flow_P_upper = pyo.Constraint(model.LC, rule=flow_P_upper_rule)
    
    # Big-M constraints for reactive power flows.
    def flow_Q_lower_rule(m, l, i, j):
        return m.Q[l, i, j] >= -m.M * m.d[l, i, j]
    model.flow_Q_lower = pyo.Constraint(model.LC, rule=flow_Q_lower_rule)
    
    def flow_Q_upper_rule(m, l, i, j):
        return m.Q[l, i, j] <= m.M * m.d[l, i, j]
    model.flow_Q_upper = pyo.Constraint(model.LC, rule=flow_Q_upper_rule)
    
    
    # Real power balance.
    def power_balance_real_rule(m, i):
        if i == 0:
            return pyo.Constraint.Skip
        incoming = sum(m.P[l, a, b] for (l, a, b) in m.LC if b == i)
        outgoing = sum(m.P[l, a, b] for (l, a, b) in m.LC if a == i)
        return incoming - outgoing == m.p_node[i]
    model.power_balance_real = pyo.Constraint(model.N, rule=power_balance_real_rule)
    
    # Reactive power balance.
    def power_balance_reactive_rule(m, i):
        if i == 0:
            return pyo.Constraint.Skip
        incoming = sum(m.Q[l, a, b] for (l, a, b) in m.LC if b == i)
        outgoing = sum(m.Q[l, a, b] for (l, a, b) in m.LC if a == i)
        return incoming - outgoing == m.q_node[i]
    model.power_balance_reactive = pyo.Constraint(model.N, rule=power_balance_reactive_rule)
    
    # Radiality constraint: each non-slack bus must have one incoming candidate.
    def radiality_rule(m, i):
        if i == 0:
            return pyo.Constraint.Skip
        return sum(m.d[l, a, b] for (l, a, b) in m.LC if b == i) == 1
    model.radiality = pyo.Constraint(model.N, rule=radiality_rule)
    
    # Objective function: minimize approximate losses weighted by line resistances.
    def objective_rule(m):
        return sum(m.r[l] * (m.P[l, a, b]**2 + m.Q[l, a, b]**2)
                   for (l, a, b) in m.LC)
    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)
    
    return model
