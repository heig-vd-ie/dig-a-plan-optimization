import pyomo.environ as pyo
from pyomo.environ import ConstraintList


def combined_model_constraints(model: pyo.AbstractModel) -> pyo.AbstractModel:
    # Radiality: each non-slack node has one incoming flow
    # model.flow_balance = pyo.Constraint(model.N, rule=flow_balance_rule)
    # model.edge_propagation = pyo.Constraint(model.L, rule=edge_propagation_rule)
    # model.upper_switch_propagation = pyo.Constraint(
    #     model.C, rule=upper_switch_propagation_rule
    # )
    # model.lower_switch_propagation = pyo.Constraint(
    #     model.C, rule=lower_switch_propagation_rule
    # )
    # model.nb_closed_switches = pyo.Constraint(rule=nb_closed_switches_rule)
    # model.parent_node = pyo.Constraint(model.C, rule=parent_node_rule)
    # model.radiality = pyo.Constraint(model.N, rule=radiality_rule)
    # model.edge_direction = pyo.Constraint(model.L, rule=edge_direction_rule)
    # DistFlow and power balance
    model.slack_voltage = pyo.Constraint(model.N, rule=slack_voltage_rule)
    model.node_active_power_balance = pyo.Constraint(
        model.N, rule=node_active_power_balance_rule
    )
    model.node_reactive_power_balance = pyo.Constraint(
        model.N, rule=node_reactive_power_balance_rule
    )
    model.edge_active_power_balance = pyo.Constraint(
        model.L, rule=edge_active_power_balance_rule
    )
    model.edge_reactive_power_balance = pyo.Constraint(
        model.L, rule=edge_reactive_power_balance_rule
    )
    model.voltage_drop_lower = pyo.Constraint(model.C, rule=voltage_drop_lower_rule)
    model.voltage_drop_upper = pyo.Constraint(model.C, rule=voltage_drop_upper_rule)
    model.current_rotated_cone = pyo.Constraint(model.C, rule=current_rotated_cone_rule)
    model.switch_active_power_lower_bound = pyo.Constraint(
        model.C, rule=switch_active_power_lower_bound_rule
    )
    model.switch_active_power_upper_bound = pyo.Constraint(
        model.C, rule=switch_active_power_upper_bound_rule
    )
    model.switch_reactive_power_lower_bound = pyo.Constraint(
        model.C, rule=switch_reactive_power_lower_bound_rule
    )
    model.switch_reactive_power_upper_bound = pyo.Constraint(
        model.C, rule=switch_reactive_power_upper_bound_rule
    )
    model.current_balance = pyo.Constraint(model.C, rule=current_balance_rule)
    # Physical limits and objective
    model.current_limit = pyo.Constraint(model.C, rule=current_limit_rule)
    model.voltage_upper_limits = pyo.Constraint(model.N, rule=voltage_upper_limits_rule)
    model.voltage_lower_limits = pyo.Constraint(model.N, rule=voltage_lower_limits_rule)
    model.objective = pyo.Objective(
        rule=objective_rule_infeasibility, sense=pyo.minimize
    )
    return model


def objective_rule_loss(m):
    # Minimize network losses
    return sum(m.r[l] * m.i_sq[l, i, j] for (l, i, j) in m.C if l not in m.S)


def objective_rule_infeasibility(m):
    # Minimize infeasibility
    return sum(m.i_sq_relax[l, i, j] for (l, i, j) in m.C) + sum(
        m.v_sq_relax_pos[n] + m.v_sq_relax_neg[n] for n in m.N
    )


def flow_balance_rule(m, n):
    # For each node, the sum of flows into the node equals the sum of flows out
    if n == m.slack_node:
        return sum(m.flow[l, i, j] for l, i, j in m.C if i == n) >= m.small_m * (
            len(m.N) - 1
        )
    else:
        return sum(m.flow[l, i, j] for l, i, j in m.C if i == n) == -m.small_m


def edge_propagation_rule(m, l):
    return sum(m.flow[l_, i, j] for l_, i, j in m.C if l_ == l) == 0


def upper_switch_propagation_rule(m, l, i, j):
    if l in m.S:
        return m.flow[l, i, j] <= m.small_m * len(m.N) * m.delta[l]
    return pyo.Constraint.Skip


def lower_switch_propagation_rule(m, l, i, j):
    if l in m.S:
        return m.flow[l, i, j] >= -m.small_m * len(m.N) * m.delta[l]
    return pyo.Constraint.Skip


def nb_closed_switches_rule(m):
    # Exactly |N|-|nS|-1 switches closed for radial network
    return sum(m.delta[l] for l in m.S) == len(m.N) - len(m.nS) - 1


def parent_node_rule(m, l, i, j):
    if i == m.slack_node:
        return m.d[l, i, j] == 0
    else:
        return pyo.Constraint.Skip


def radiality_rule(m, n):
    if n != m.slack_node:
        return sum(m.d[l, i, j] for l, i, j in m.C if i == n) == 1
    else:
        return pyo.Constraint.Skip


def edge_direction_rule(m, l):
    if l in m.S:
        return sum(m.d[l_, i, j] for l_, i, j in m.C if l_ == l) == m.delta[l]
    else:
        return pyo.Constraint.Skip


def slack_voltage_rule(m, n):
    if n == m.slack_node:
        return m.v_sq[n] == m.slack_node_v_sq
    return pyo.Constraint.Skip


def node_active_power_balance_rule(m, n):
    p_flow_tot = sum(m.p_flow[l, i, j] for l, i, j in m.C if i == n)
    if n == m.slack_node:
        return p_flow_tot == -m.p_slack_node
    return p_flow_tot == -m.p_node[n]


def node_reactive_power_balance_rule(m, n):
    q_flow_tot = sum(
        m.q_flow[l, i, j] - m.b[l] / 2 * m.v_sq[i] for l, i, j in m.C if i == n
    )
    if n == m.slack_node:
        return q_flow_tot == -m.q_slack_node
    return q_flow_tot == -m.q_node[n]


def edge_active_power_balance_rule(m, l):
    if l in m.S:
        return sum(m.p_flow[l_, i, j] for l_, i, j in m.C if l_ == l) == 0
    return (
        sum(
            (m.p_flow[l_, i, j] - m.r[l_] / 2 * m.i_sq[l_, i, j])
            for l_, i, j in m.C
            if l_ == l
        )
        == 0
    )


def edge_reactive_power_balance_rule(m, l):
    if l in m.S:
        return sum(m.q_flow[l_, i, j] for l_, i, j in m.C if l_ == l) == 0
    return (
        sum(
            (m.q_flow[l_, i, j] - m.x[l_] / 2 * m.i_sq[l_, i, j])
            for l_, i, j in m.C
            if l_ == l
        )
        == 0
    )


def current_balance_rule(m, l, i, j):
    return m.i_sq[l, i, j] == m.i_sq[l, j, i]


def current_rotated_cone_rule(m, l, i, j):
    if l in m.S:
        return pyo.Constraint.Skip
    else:
        lhs = (m.p_flow[l, i, j]) ** 2 + (m.q_flow[l, i, j]) ** 2
        rhs = m.i_sq[l, i, j] * m.v_sq[i] / (m.n_transfo[l, i, j] ** 2)
        return lhs <= rhs


def voltage_drop_lower_rule(m, l, i, j):
    if l in m.S:
        return m.v_sq[i] - m.v_sq[j] >= -m.big_m * (1 - m.delta[l])
    else:
        dv = (
            -2 * (m.r[l] * m.p_flow[l, i, j] + m.x[l] * m.q_flow[l, i, j])
            + (m.r[l] ** 2 + m.x[l] ** 2) * m.i_sq[l, i, j]
        )
        voltage_diff = (
            m.v_sq[i] / (m.n_transfo[l, i, j] ** 2)
            - m.v_sq[j] / (m.n_transfo[l, j, i] ** 2)
            + dv
        )
        return voltage_diff == 0


def voltage_drop_upper_rule(m, l, i, j):
    if l in m.S:
        return m.v_sq[i] - m.v_sq[j] <= m.big_m * (1 - m.delta[l])
    else:
        return pyo.Constraint.Skip


def switch_active_power_lower_bound_rule(m, l, i, j):
    if l in m.S:
        return m.p_flow[l, i, j] >= -m.big_m * m.delta[l]
    else:
        return pyo.Constraint.Skip


def switch_active_power_upper_bound_rule(m, l, i, j):
    if l in m.S:
        return m.p_flow[l, i, j] <= m.big_m * m.delta[l]
    else:
        return pyo.Constraint.Skip


def switch_reactive_power_lower_bound_rule(m, l, i, j):
    if l in m.S:
        return m.q_flow[l, i, j] >= -m.big_m * m.delta[l]
    else:
        return pyo.Constraint.Skip


def switch_reactive_power_upper_bound_rule(m, l, i, j):
    if l in m.S:
        return m.q_flow[l, i, j] <= m.big_m * m.delta[l]
    else:
        return pyo.Constraint.Skip


def current_limit_rule(m, l, i, j):
    if l in m.S:
        return pyo.Constraint.Skip
    else:
        return m.i_sq[l, i, j] <= m.i_max[l] ** 2 + m.i_sq_relax[l, i, j]


def voltage_upper_limits_rule(m, n):
    return m.v_sq[n] <= m.v_max[n] ** 2 + m.v_sq_relax_pos[n]


def voltage_lower_limits_rule(m, n):
    return m.v_sq[n] >= m.v_min[n] ** 2 - m.v_sq_relax_neg[n]
