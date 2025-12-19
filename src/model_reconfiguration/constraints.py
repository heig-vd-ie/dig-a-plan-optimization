#### OBJECTIVE FUNCTIONS ####


def master_obj(m):
    return m.θ1 + m.θ2


def objective_rule_loss(m):
    # Minimize network losses
    return sum(m.r[l] * m.i_sq[l, i, j, ω] for (l, i, j, ω) in m.ClΩ) + sum(
        m.r[l] * m.i_sq[l, i, j, ω] * m.γ_trafo_loss for (l, i, j, ω) in m.CtΩ
    )


def objective_rule_infeasibility(m):
    p_curt = sum(m.p_curt_cons[n, ω] + m.p_curt_prod[n, ω] for n in m.N for ω in m.Ω)
    q_curt = sum(m.q_curt_cons[n, ω] + m.q_curt_prod[n, ω] for n in m.N for ω in m.Ω)
    v_relax = sum(m.v_relax_up[n, ω] + m.v_relax_down[n, ω] for n, ω in m.NΩ)

    return p_curt + q_curt + v_relax


def objective_rule_admm_penalty(m):
    return (
        (m.ρ / 2.0) * sum((m.δ[s] - m.zδ[s]) ** 2 for s in m.S)
        + sum(m.λδ[s] * m.δ[s] for s in m.S)
        + (m.ρ / 2.0) * sum((m.ζ[tr, tap] - m.zζ[tr, tap]) ** 2 for tr, tap in m.TrTaps)
        + sum(m.λζ[tr, tap] * m.ζ[tr, tap] for tr, tap in m.TrTaps)
    )


def objective_rule_output(m):
    return sum(
        m.p_curt_cons[n, ω] * m.voll + m.p_curt_prod[n, ω] * m.volp
        for n in m.N
        for ω in m.Ω
    )


def objective_rule_combined(m):
    # Minimize network losses and infeasibility penalties
    return (
        objective_rule_loss(m)
        + objective_rule_infeasibility(m) * m.γ_infeasibility
        + objective_rule_admm_penalty(m) * m.γ_admm_penalty
    )


##### CONSTRAINTS #####


# Radiality constraint: each non-slack bus must have one incoming candidate.
def imaginary_flow_balance_slack_rule(m, n):
    return sum(m.flow[l, i, j] for l, i, j in m.C if i == n) >= m.ε * (len(m.N) - 1)


def imaginary_flow_balance_rule(m, n):
    return sum(m.flow[l, i, j] for l, i, j in m.C if i == n) == -m.ε


# Orientation constraint.
def imaginary_flow_edge_propagation_rule(m, l):
    return sum(m.flow[l_, i, j] for l_, i, j in m.C if l_ == l) == 0


def imaginary_flow_upper_switch_propagation_rule(m, l, i, j):
    return m.flow[l, i, j] <= m.ε * len(m.N) * m.δ[l]


def imaginary_flow_lower_switch_propagation_rule(m, l, i, j):
    return m.flow[l, i, j] >= -m.ε * len(m.N) * m.δ[l]


def imaginary_flow_nb_closed_switches_rule(m):
    return sum(m.δ[l] for l in m.S) == len(m.N) - m.number_of_lines - 1


def radiality_rule(m, n):
    return sum(m.d[l, i, j] for l, i, j in m.C if i == n) == 1


def edge_direction_rule(m, l):
    return sum(m.d[l_, i, j] for l_, i, j in m.C if l_ == l) == m.δ[l]


def master_switch_status_propagation_rule(m, s):
    return m.δ[s] == m.master_δ[s]


def master_transformer_status_propagation_rule(m, tr, tap):
    return m.ζ[tr, tap] == m.master_ζ[tr, tap]


# (1) Slack Bus: fix bus 0's voltage squared to 1.0.


def slack_voltage_rule(m, n, ω):
    return m.v_sq[n, ω] == m.slack_node_v_sq[ω]  # Slack bus voltage squared


# (2) Node Power Balance (Real) for candidate (l,i,j).
# For candidate (l, i, j), j is the downstream bus.


def node_active_power_balance_slack_rule(m, n, ω):
    return (
        sum(m.p_flow[l, i, j, ω] for (l, i, j) in m.C if (i == n)) == -m.p_slack_node[ω]
    )


def node_active_power_balance_rule(m, n, ω):
    return (
        sum(m.p_flow[l, i, j, ω] for (l, i, j) in m.C if (i == n))
        == m.p_node_prod[n, ω] * m.node_prod_installed[n, ω]
        - m.p_node_cons[n, ω] * m.node_cons_installed[n, ω]
        + m.p_curt_cons[n, ω]
        - m.p_curt_prod[n, ω]
    )


def node_active_power_rule(m, n, ω):
    return m.p_curt_cons[n, ω] <= m.p_node_cons[n, ω] * m.node_cons_installed[n, ω]


def node_active_power_prod_rule(m, n, ω):
    return m.p_curt_prod[n, ω] <= m.p_node_prod[n, ω] * m.node_prod_installed[n, ω]


# (3) Node Power Balance (Reactive) for candidate (l,i,j).
def node_reactive_power_balance_slack_rule(m, n, ω):
    return (
        sum(
            m.q_flow[l, i, j, ω] - m.b[l] / 2 * m.v_sq[i, ω]
            for (l, i, j) in m.C
            if (i == n)
        )
        == -m.q_slack_node[ω]
    )


def node_reactive_power_balance_rule(m, n, ω):
    return (
        sum(
            m.q_flow[l, i, j, ω] - m.b[l] / 2 * m.v_sq[i, ω]
            for (l, i, j) in m.C
            if (i == n)
        )
        == m.q_node_prod[n, ω] * m.node_prod_installed[n, ω]
        - m.q_node_cons[n, ω] * m.node_cons_installed[n, ω]
        + m.q_curt_cons[n, ω]
        - m.q_curt_prod[n, ω]
    )


def node_reactive_power_rule(m, n, ω):
    return m.q_curt_cons[n, ω] <= m.q_node_cons[n, ω] * m.node_cons_installed[n, ω]


def node_reactive_power_prod_rule(m, n, ω):
    return m.q_curt_prod[n, ω] <= m.q_node_prod[n, ω] * m.node_prod_installed[n, ω]


def installed_prod_rule(m, n, ω):
    return m.node_prod_installed[n, ω] == m.node_prod_installed_param[n]


def installed_cons_rule(m, n, ω):
    return m.node_cons_installed[n, ω] == m.node_cons_installed_param[n]


def edge_active_power_balance_switch_rule(m, l, ω):
    return sum(m.p_flow[l_, i, j, ω] for (l_, i, j) in m.C if l_ == l) == 0


def edge_active_power_balance_line_rule(m, l, ω):
    return (
        sum(
            m.p_flow[l_, i, j, ω] - m.r[l_] / 2 * m.i_sq[l_, i, j, ω]
            for (l_, i, j) in m.C
            if l_ == l
        )
        == 0
    )


def edge_reactive_power_balance_switch_rule(m, l, ω):
    return sum(m.q_flow[l_, i, j, ω] for (l_, i, j) in m.C if l_ == l) == 0


def edge_reactive_power_balance_line_rule(m, l, ω):
    return (
        sum(
            m.q_flow[l_, i, j, ω] - m.x[l_] / 2 * m.i_sq[l_, i, j, ω]
            for (l_, i, j) in m.C
            if l_ == l
        )
        == 0
    )


def edge_active_power_balance_lindistflow_rule(m, l, ω):
    return sum(m.p_flow[l_, i, j, ω] for (l_, i, j) in m.C if l_ == l) == 0


def edge_reactive_power_balance_lindistflow_rule(m, l, ω):
    return sum(m.q_flow[l_, i, j, ω] for (l_, i, j) in m.C if l_ == l) == 0


def current_balance_rule(m, l, i, j, ω):
    return (
        m.i_sq[l, i, j, ω] == m.i_sq[l, j, i, ω]
    )  # Ensure current balance in both directions


def current_rotated_cone_rule_transformed(m, l, i, j, ω):
    lhs = (
        (2 * m.p_flow[l, i, j, ω]) ** 2
        + (2 * m.q_flow[l, i, j, ω]) ** 2
        + (m.v_sq[i, ω] - m.i_sq[l, i, j, ω]) ** 2
    )
    rhs = (m.v_sq[i, ω] + m.i_sq[l, i, j, ω]) ** 2

    return lhs <= rhs


def current_rotated_cone_rule_transformer_transformed(m, l, i, j, ω):
    lhs = (
        (2 * m.p_flow[l, i, j, ω]) ** 2
        + (2 * m.q_flow[l, i, j, ω]) ** 2
        + (m.v_sq[i, ω] - m.i_sq[l, i, j, ω]) ** 2
    )
    rhs = (m.v_sq[i, ω] + m.i_sq[l, i, j, ω]) ** 2

    return lhs <= rhs


def current_rotated_cone_rule(m, l, i, j, ω):
    lhs = (m.p_flow[l, i, j, ω]) ** 2 + (m.q_flow[l, i, j, ω]) ** 2
    rhs = m.i_sq[l, i, j, ω] * m.v_sq[i, ω]
    return lhs <= rhs


def current_rotated_cone_transformer_rule(m, l, i, j, ω):
    lhs = (m.p_flow[l, i, j, ω]) ** 2 + (m.q_flow[l, i, j, ω]) ** 2
    rhs = m.i_sq[l, i, j, ω] * m.v_sq[i, ω]
    return lhs <= rhs


# (4) Voltage Drop along Branch for candidate (l,i,j).


def voltage_limit_lower_rule(m, l, i, j, ω):
    return m.v_sq[i, ω] - m.v_sq[j, ω] <= m.big_m * (1 - m.δ[l])


def voltage_limit_upper_rule(m, l, i, j, ω):
    return m.v_sq[i, ω] - m.v_sq[j, ω] >= -m.big_m * (1 - m.δ[l])


def voltage_drop_line_rule(m, l, i, j, ω):
    dv = (
        -2 * (m.r[l] * m.p_flow[l, i, j, ω] + m.x[l] * m.q_flow[l, i, j, ω])
        + (m.r[l] ** 2 + m.x[l] ** 2) * m.i_sq[l, i, j, ω]
    )
    voltage_diff = m.v_sq[i, ω] - m.v_sq[j, ω] + dv
    return voltage_diff == 0


def voltage_drop_line_lindistflow_rule(m, l, i, j, ω):
    dv = -2 * (m.r[l] * m.p_flow[l, i, j, ω] + m.x[l] * m.q_flow[l, i, j, ω])
    voltage_diff = m.v_sq[i, ω] - m.v_sq[j, ω] + dv
    return voltage_diff == 0


def voltage_drop_transfo_rule(m, l, i, j, ω):
    dv = (
        -2 * (m.r[l] * m.p_flow[l, i, j, ω] + m.x[l] * m.q_flow[l, i, j, ω])
        + (m.r[l] ** 2 + m.x[l] ** 2) * m.i_sq[l, i, j, ω]
    )
    if i > j:
        voltage_diff = m.v_sq[i, ω] - m.v_sq[j, ω] + dv + m.vt_sq[i, ω]
    else:
        voltage_diff = m.v_sq[i, ω] - m.v_sq[j, ω] + dv - m.vt_sq[j, ω]
    return voltage_diff == 0


def voltage_drop_transfo_lindistflow_rule(m, tr, i, j, ω):
    dv = -2 * (m.r[tr] * m.p_flow[tr, i, j, ω] + m.x[tr] * m.q_flow[tr, i, j, ω])
    voltage_diff = m.v_sq[i, ω] - m.v_sq[j, ω] + dv
    return voltage_diff == 0


def voltage_tap_upper_limit_rule(m, tr, i, tap, ω):
    return m.vt_sq[i, ω] <= m.v_sq[i, ω] * (tap - 100) / 100 + 10 * (1 - m.ζ[tr, tap])


def voltage_tap_lower_limit_rule(m, tr, i, tap, ω):
    return m.vt_sq[i, ω] >= m.v_sq[i, ω] * (tap - 100) / 100 - 10 * (1 - m.ζ[tr, tap])


def tap_limit_rule(m, tr):
    return sum(m.ζ[tr, tap] for tr0, tap in m.TrTaps if tr0 == tr) == 1


def switch_active_power_lower_bound_rule(m, l, i, j, ω):
    return m.p_flow[l, i, j, ω] >= -m.big_m * m.δ[l]


def switch_active_power_upper_bound_rule(m, l, i, j, ω):
    return m.p_flow[l, i, j, ω] <= m.big_m * m.δ[l]


def switch_reactive_power_lower_bound_rule(m, l, i, j, ω):
    return m.q_flow[l, i, j, ω] >= -m.big_m * m.δ[l]


def switch_reactive_power_upper_bound_rule(m, l, i, j, ω):
    return m.q_flow[l, i, j, ω] <= m.big_m * m.δ[l]


def optimal_current_limit_rule(m, l, i, j, ω):
    return m.i_sq[l, i, j, ω] <= m.i_max[l] ** 2


def optimal_voltage_upper_limits_rule(m, n, ω):
    return m.v_sq[n, ω] <= m.v_max[n] ** 2 + m.v_relax_up[n, ω]


def optimal_voltage_lower_limits_rule(m, n, ω):
    return m.v_sq[n, ω] >= m.v_min[n] ** 2 - m.v_relax_down[n, ω]


def optimal_voltage_upper_limits_distflow_rule(m, n, ω):
    return m.v_sq[n, ω] <= m.slack_node_v_sq[ω] + 0.05


def optimal_voltage_lower_limits_distflow_rule(m, n, ω):
    return m.v_sq[n, ω] >= m.slack_node_v_sq[ω] - 0.05
