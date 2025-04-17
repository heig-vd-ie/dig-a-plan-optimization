
import pyomo.environ as pyo
from set_model_input import build_common_components

def master_model():
    """
    Creates a Pyomo AbstractModel for network reconfiguration using a candidate set LF.
    
    Sets:
      - I: set of buses.
      - L: set of physical lines.
      - F: for each line l in L, F[l] should be supplied as a collection of candidate bus pairs,
           for example: F[l] = {(i, j), (j, i)}.
      - LF: a flat candidate set defined as:
              LF = { (l, i, j) for each l in L and for each (i,j) in F[l] }.
    
    Decision Variables:
      - d[l, i, j]: binary variable that indicates whether candidate orientation (i,j) is selected for line l.
      - Delta[l]: binary switch variable for switchable lines (l in S).
      - P[l, i, j] and Q[l, i, j]: flow variables (real and reactive, respectively) for candidate orientation (i, j) of line l.
      - v_sq[i]: squared voltage at bus i.
      
    Constraints:
      - Orientation: For each line l, the sum over candidates in LF (filtered for l) equals 1 for non-switchable lines, or equals Delta[l] for switchable lines.
      - Big-M constraints on flows enforce that if a candidate orientation is not selected (d = 0), its associated flows are forced toward zero.
      - Voltage limits, a fixed slack bus, real and reactive power balance, and radiality constraints are imposed.
      
    Objective:
      Minimize approximate losses computed from the candidate flows weighted by line resistances.
    """

    model = pyo.AbstractModel()
    
    # Build common parts
    build_common_components(model)
    
    # === Master Stage Decision Variables ===
    # Orientation binary variable over candidates (model.LF)
    model.d = pyo.Var(model.LF, domain=pyo.Binary)
    # Switch variable for switchable lines.
    model.Delta = pyo.Var(model.S, domain=pyo.Binary)
    # Flow variables (real and reactive) defined over the candidate set
    model.P = pyo.Var(model.LF, domain=pyo.Reals)
    model.Q = pyo.Var(model.LF, domain=pyo.Reals)
    # # Bus voltage squared.
    # model.v_sq = pyo.Var(model.I, domain=pyo.NonNegativeReals)
    
    # Orientation constraint:
    # For each line l, sum over all candidates (l, i, j) in model.LF (with l fixed) equals:
    #    1, if line l is non-switchable; otherwise, equals Delta[l].
    def orientation_rule(m, l):
        if l in m.S:
            return sum(m.d[l_, i, j] for (l_, i, j) in m.LF if l_ == l) == m.Delta[l]
        else:
            return sum(m.d[l_, i, j] for (l_, i, j) in m.LF if l_ == l) == 1
    model.orientation = pyo.Constraint(model.L, rule=orientation_rule)
    
    # Big-M constraints for real power flows.
    def flow_P_lower_rule(m, l, i, j):
        return m.P[l, i, j] >= -m.M * m.d[l, i, j]
    model.flow_P_lower = pyo.Constraint(model.LF, rule=flow_P_lower_rule)
    
    def flow_P_upper_rule(m, l, i, j):
        return m.P[l, i, j] <= m.M * m.d[l, i, j]
    model.flow_P_upper = pyo.Constraint(model.LF, rule=flow_P_upper_rule)
    
    # Big-M constraints for reactive power flows.
    def flow_Q_lower_rule(m, l, i, j):
        return m.Q[l, i, j] >= -m.M * m.d[l, i, j]
    model.flow_Q_lower = pyo.Constraint(model.LF, rule=flow_Q_lower_rule)
    
    def flow_Q_upper_rule(m, l, i, j):
        return m.Q[l, i, j] <= m.M * m.d[l, i, j]
    model.flow_Q_upper = pyo.Constraint(model.LF, rule=flow_Q_upper_rule)
    
    # # Slack bus voltage: fix bus 0 to 1.0.
    # def slack_voltage_rule(m):
    #     return m.v_sq[0] == 1.0
    # model.slack_voltage = pyo.Constraint(rule=slack_voltage_rule)
    
    # # Voltage limits: for each bus i, ensure 0.9^2 <= v_sq[i] <= 1.1^2.
    # vmin_sq = 0.9**2
    # vmax_sq = 1.1**2
    # def voltage_limits_rule(m, i):
    #     return pyo.inequality(vmin_sq, m.v_sq[i], vmax_sq)
    # model.voltage_limits = pyo.Constraint(model.I, rule=voltage_limits_rule)
    
    # Real power balance:
    # For each bus i (except the slack bus 0), the net real power is the sum of flows from candidates with head i (incoming)
    # minus the sum of flows from candidates with tail i (outgoing) and must equal p_load[i].
    def power_balance_real_rule(m, i):
        if i == 0:
            return pyo.Constraint.Skip
        incoming = sum(m.P[l, a, b] for (l, a, b) in m.LF if b == i)
        outgoing = sum(m.P[l, a, b] for (l, a, b) in m.LF if a == i)
        return incoming - outgoing == m.p_load[i]
    model.power_balance_real = pyo.Constraint(model.I, rule=power_balance_real_rule)
    
    # Reactive power balance:
    def power_balance_reactive_rule(m, i):
        if i == 0:
            return pyo.Constraint.Skip
        incoming = sum(m.Q[l, a, b] for (l, a, b) in m.LF if b == i)
        outgoing = sum(m.Q[l, a, b] for (l, a, b) in m.LF if a == i)
        return incoming - outgoing == m.q_load[i]
    model.power_balance_reactive = pyo.Constraint(model.I, rule=power_balance_reactive_rule)
    
    # Radiality constraint:
    # Each non-slack bus i must have exactly one candidate orientation delivering power into it.
    def radiality_rule(m, i):
        if i == 0:
            return pyo.Constraint.Skip
        return sum(m.d[l, a, b] for (l, a, b) in m.LF if b == i) == 1
    model.radiality = pyo.Constraint(model.I, rule=radiality_rule)
    
    # Objective: minimize approximate losses (sum over all candidate flows weighted by the line resistances).
    def objective_rule(m):
        return sum(m.r[l] * (m.P[l, a, b]**2 + m.Q[l, a, b]**2)
                   for (l, a, b) in m.LF)
    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)
    
    return model

