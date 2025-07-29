import pyomo.environ as pyo

#### OBJECTIVE FUNCTIONS ####


def master_obj(m):
    return m.theta


def objective_rule_loss(m):
    # Minimize network losses
    return sum(m.r[l] * m.i_sq[l, i, j] for (l, i, j) in m.C if l not in m.S)


def objective_rule_infeasibility(m):
    v_penalty = sum(m.slack_v_pos[n] + m.slack_v_neg[n] for n in m.N)
    i_penalty = sum(m.slack_i_sq[l, i, j] for (l, i, j) in m.C)

    return v_penalty + i_penalty


def objective_rule_penalty(m):
    return sum(m.δ_penalty[l] for l in m.S)


def objective_rule_admm_penalty(m):
    return sum(m.λ[s] * m.δ[s] for s in m.S) + (m.ρ / 2.0) * sum(
        (m.δ[s] - m.z[s]) ** 2 for s in m.S
    )


def objective_rule_combined(m):
    # Minimize network losses and infeasibility penalties
    return (
        objective_rule_loss(m)
        + objective_rule_infeasibility(m) * m.weight_infeasibility
        + objective_rule_penalty(m) * m.weight_penalty
        + objective_rule_admm_penalty(m) * m.weight_admm_penalty
    )


##### CONSTRAINTS #####


# Radiality constraint: each non-slack bus must have one incoming candidate.
def imaginary_flow_balance_rule(m, n):
    if n == m.slack_node:
        return sum(m.flow[l, i, j] for l, i, j in m.C if i == n) >= m.small_m * (
            len(m.N) - 1
        )
    else:
        return sum(m.flow[l, i, j] for l, i, j in m.C if i == n) == -m.small_m


# Orientation constraint.
def imaginary_flow_edge_propagation_rule(m, l):
    return sum(m.flow[l_, i, j] for l_, i, j in m.C if l_ == l) == 0


def imaginary_flow_upper_switch_propagation_rule(m, l, i, j):
    if l in m.S:
        return m.flow[l, i, j] <= m.small_m * len(m.N) * m.δ[l]
    else:
        return pyo.Constraint.Skip


def imaginary_flow_lower_switch_propagation_rule(m, l, i, j):
    if l in m.S:
        return m.flow[l, i, j] >= -m.small_m * len(m.N) * m.δ[l]
    else:
        return pyo.Constraint.Skip


def imaginary_flow_nb_closed_switches_rule(m):
    return sum(m.δ[l] for l in m.S) == len(m.N) - len(m.nS) - 1


def radiality_rule(m, n):
    if n != m.slack_node:
        return sum(m.d[l, i, j] for l, i, j in m.C if i == n) == 1
    else:
        return pyo.Constraint.Skip


def edge_direction_rule(m, l):
    if l in m.S:
        return sum(m.d[l_, i, j] for l_, i, j in m.C if l_ == l) == m.δ[l]
    else:
        return pyo.Constraint.Skip


def master_switch_status_propagation_rule(m, s):
    return m.δ[s] == m.master_δ[s]


# (1) Slack Bus: fix bus 0's voltage squared to 1.0.


def slack_voltage_rule(m, n):
    if n == m.slack_node:
        return m.v_sq[n] == m.slack_node_v_sq
    return pyo.Constraint.Skip


# (2) Node Power Balance (Real) for candidate (l,i,j).
# For candidate (l, i, j), j is the downstream bus.


def node_active_power_balance_rule(m, n):
    p_flow_tot = sum(m.p_flow[l, i, j] for (l, i, j) in m.C if (i == n))
    if n == m.slack_node:
        return p_flow_tot == -m.p_slack_node
    else:
        return p_flow_tot == -m.p_node[n]


# (3) Node Power Balance (Reactive) for candidate (l,i,j).
def node_reactive_power_balance_rule(m, n):
    q_flow_tot = sum(
        m.q_flow[l, i, j] - m.b[l] / 2 * m.v_sq[i] for (l, i, j) in m.C if (i == n)
    )
    if n == m.slack_node:
        return q_flow_tot == -m.q_slack_node
    else:
        return q_flow_tot == -m.q_node[n]


def edge_active_power_balance_rule(m, l):
    if l in m.S:
        return sum(m.p_flow[l_, i, j] for (l_, i, j) in m.C if l_ == l) == 0
    else:
        return (
            sum(
                m.p_flow[l_, i, j] - m.r[l_] / 2 * m.i_sq[l_, i, j]
                for (l_, i, j) in m.C
                if l_ == l
            )
            == 0
        )


def edge_reactive_power_balance_rule(m, l):
    if l in m.S:
        return sum(m.q_flow[l_, i, j] for (l_, i, j) in m.C if l_ == l) == 0
    else:
        return (
            sum(
                m.q_flow[l_, i, j] - m.x[l_] / 2 * m.i_sq[l_, i, j]
                for (l_, i, j) in m.C
                if l_ == l
            )
            == 0
        )


def edge_active_power_balance_lindistflow_rule(m, l):
    return sum(m.p_flow[l_, i, j] for (l_, i, j) in m.C if l_ == l) == 0


def edge_reactive_power_balance_lindistflow_rule(m, l):
    return sum(m.q_flow[l_, i, j] for (l_, i, j) in m.C if l_ == l) == 0


def current_balance_rule(m, l, i, j):
    return (
        m.i_sq[l, i, j] == m.i_sq[l, j, i]
    )  # Ensure current balance in both directions


def current_rotated_cone_rule_transformed(m, l, i, j):
    if l in m.S:
        return pyo.Constraint.Skip
    else:

        lhs = (
            (2 * m.p_flow[l, i, j]) ** 2
            + (2 * m.q_flow[l, i, j]) ** 2
            + (m.v_sq[i] / (m.n_transfo[l, i, j] ** 2) - m.i_sq[l, i, j]) ** 2
        )
        rhs = (m.v_sq[i] / (m.n_transfo[l, i, j] ** 2) + m.i_sq[l, i, j]) ** 2

        return lhs <= rhs


def current_rotated_cone_rule(m, l, i, j):
    if l in m.S:
        return pyo.Constraint.Skip
    else:

        lhs = (m.p_flow[l, i, j]) ** 2 + (m.q_flow[l, i, j]) ** 2
        rhs = m.i_sq[l, i, j] * m.v_sq[i] / (m.n_transfo[l, i, j] ** 2)
        return lhs <= rhs


# (4) Voltage Drop along Branch for candidate (l,i,j).
# Let expr = v_sq[i] - 2*(r[l]*p_z_up(l,i,j) + x[l]*q_z_up(l,i,j)) + (r[l]^2+x[l]^2)*f_c(l,i,j).
# We then enforce two separate inequalities


def voltage_drop_upper_rule(m, l, i, j):
    if l in m.S:
        return m.v_sq[i] - m.v_sq[j] <= m.big_m * (1 - m.δ[l])
    else:
        return pyo.Constraint.Skip


def voltage_drop_lower_rule(m, l, i, j):
    if l in m.S:
        return m.v_sq[i] - m.v_sq[j] >= -m.big_m * (1 - m.δ[l])
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


def voltage_drop_upper_lindistflow_rule(m, l, i, j):
    if l in m.S:
        return m.v_sq[i] - m.v_sq[j] <= m.big_m * (1 - m.δ[l])
    else:
        return pyo.Constraint.Skip


def voltage_drop_lower_lindistflow_rule(m, l, i, j):
    if l in m.S:
        return m.v_sq[i] - m.v_sq[j] >= -m.big_m * (1 - m.δ[l])
    else:
        dv = -2 * (m.r[l] * m.p_flow[l, i, j] + m.x[l] * m.q_flow[l, i, j])
        voltage_diff = (
            m.v_sq[i] / (m.n_transfo[l, i, j] ** 2)
            - m.v_sq[j] / (m.n_transfo[l, j, i] ** 2)
            + dv
        )
        return voltage_diff == 0


def switch_active_power_lower_bound_rule(m, l, i, j):
    if l in m.S:
        return m.p_flow[l, i, j] >= -m.big_m * m.δ[l]
    else:
        return pyo.Constraint.Skip


def switch_active_power_upper_bound_rule(m, l, i, j):
    if l in m.S:
        return m.p_flow[l, i, j] <= m.big_m * m.δ[l]
    else:
        return pyo.Constraint.Skip


def switch_reactive_power_lower_bound_rule(m, l, i, j):
    if l in m.S:
        return m.q_flow[l, i, j] >= -m.big_m * m.δ[l]
    else:
        return pyo.Constraint.Skip


def switch_reactive_power_upper_bound_rule(m, l, i, j):
    if l in m.S:
        return m.q_flow[l, i, j] <= m.big_m * m.δ[l]
    else:
        return pyo.Constraint.Skip


def optimal_current_limit_rule(m, l, i, j):
    if l in m.S:
        return pyo.Constraint.Skip
    else:
        return m.i_sq[l, i, j] <= m.i_max[l] ** 2


def optimal_voltage_upper_limits_rule(m, n):
    return m.v_sq[n] <= m.v_max[n] ** 2


def optimal_voltage_lower_limits_rule(m, n):
    return m.v_sq[n] >= m.v_min[n] ** 2


def optimal_voltage_upper_limits_distflow_rule(m, n):
    return m.v_sq[n] <= m.slack_node_v_sq + 0.05


def optimal_voltage_lower_limits_distflow_rule(m, n):
    return m.v_sq[n] >= m.slack_node_v_sq - 0.05


# (6) Flow Bounds for candidate (l,i,j):
def infeasible_current_limit_rule(m, l, i, j):
    if l in m.S:
        return pyo.Constraint.Skip
    else:
        return m.i_sq[l, i, j] <= m.i_max[l] ** 2 + m.slack_i_sq[l, i, j]


# (7) Voltage Limits: enforce v_sq[i] in [vmin^2, vmax^2].
def infeasible_voltage_upper_limits_rule(m, n):
    return m.v_sq[n] <= m.v_max[n] ** 2 + m.slack_v_pos[n]


def infeasible_voltage_lower_limits_rule(m, n):
    return m.v_sq[n] >= m.v_min[n] ** 2 - m.slack_v_neg[n]
