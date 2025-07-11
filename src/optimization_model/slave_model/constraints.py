r"""
2.4.1. Initialization
~~~~~~~~~~~~~~~~~~~~~

The **slave model** solves a **DistFlow optimization problem** for a fixed network topology (as determined by the master model).

The topology is passed via the binary parameter :math:`d^{\text{master}}_{l~i~j}`:


- In the **master model**, :math:`d_{l~i~j}` is a binary variable indicating whether candidate branch :math:`(l~i~j)` is active.
- In the **slave model**, :math:`d^{\text{master}}_{l~i~j}` is a fixed parameter based on the master's decision.
- When :math:`d^{\text{master}}_{l~i~j} = 0`, the branch is disabled (power flow = 0, voltage drop and SOC constraints deactivated).

This structure enables decomposition into:

- **Master**: network planning (topology)
- **Slave**: feasibility and cost evaluation via power flow analysis


2.4.2. Objective: Minimize Losses and Slack Penalties constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The objective minimizes line losses and penalizes voltage and current violations:

.. math::
    :label: distflow-objective
    :nowrap:

    \begin{align}
        \text{Objective} =\ 
        \sum_{l~i~j} r_l \cdot f^{\text{current}} \cdot i_{l~i~j}\ 
        + c^{\text{penalty}} \cdot \left( \sum_{n} (V_n^{+} + V_n^{-}) + \sum_{l~i~j} i_{l~i~j}^{\text{slack}} \right)
    \end{align}

Where:

- :math:`f^{\text{current}}` scales the squared current to model Joule losses.

- :math:`c^{\text{penalty}}` weights slack penalties to discourage limit violations.

- :math:`V_n^{+}`, :math:`V_n^{-}`: slack variables for over/under-voltage.

- :math:`i_{l~i~j}^{\text{slack}}` : slack variable for current constraint violation.



2.4.3. Slack bus constraint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The slack bus (or reference bus) is used to set the overall voltage level of the network. This constraint fixes the squared voltage at the slack node to a predetermined value:


.. math::
    :label: distflow-slack
    :nowrap:

    \begin{align}
        f^{\text{voltage}} \cdot v_n = v_{\text{slack}} \quad \text{if } n = n_{\text{slack}}
    \end{align}


2.4.4. Real Power Balance (Active Power) constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This constraint enforces Kirchhoff’s Current Law (KCL) for **active power** at each downstream node:


.. math::
    :label: distflow-power
    :nowrap:

    \begin{align}
        f^{\text{power}} \cdot p_{l~i~j} = d^{\text{master}}_{l~i~j} \cdot \left( - p_i^{\text{node}} - \sum_{l',i',j': j'=i, i'\ne j} \left( r_{l'} \cdot f^{\text{current}} \cdot i_{l'i'j'} - f^{\text{power}} \cdot p_{l'i'j'} \right) \right)
    \end{align}
    
- Ensures that incoming power equals outgoing power + local demand.
- Accumulates downstream real power and resistive losses.
- Deactivated when :math:`d^{\text{master}}_{lij} = 0`.
- :math:`f^{\text{power}}` scales real and reactive power variables in downstream branches.
- :math:`f^{\text{current}}` scales squared current terms.

2.4.5. Reactive Power Balance constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Similar to real power, this enforces reactive power consistency at downstream buses:

.. math::
    :label: distflow-power1
    :nowrap:

    \begin{align}
        f^{\text{power}} \cdot q_{l~i~j} = d^{\text{master}}_{l~i~j} \cdot \left( - q_i^{\text{node}} - \sum_{l',i',j': j'=i, i'\ne j} \left( x_{l'} \cdot f^{\text{current}} \cdot i_{l'i'j'} - f^{\text{power}} \cdot q_{l'i'j'} \right) + \sum_{l',i'} \frac{b_{l'}}{2} \cdot f^{\text{voltage}} \cdot v_i \right)
    \end{align}

- The shunt susceptance effects are account at node :math:`i`.
- :math:`f^{\text{power}}` and :math:`f^{\text{current}}` are scaling constants.
- :math:`f^{\text{voltage}}` rescales the node voltages.

2.4.6. Voltage Drop constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Models voltage drop along candidate branch :math:`(l, i, j)` with relaxation using a Big-M formulation:

.. math::
    :label: voltage-drop
    :nowrap:

    \begin{align}
        \frac{f^{\text{voltage}} \cdot v_i}{\tau_{l~i~j}^2} - \frac{f^{\text{voltage}} \cdot v_j}{\tau_{l~j~i}^2} \le \text{drop}_{l~i~j} + M(1 - d^{\text{master}}_{l~i~j})
    \end{align}

.. math:: 
    :label: voltage-drop1
    :nowrap:
    
    \begin{align}
        \frac{f^{\text{voltage}} \cdot v_i}{\tau_{l~i~j}^2} - \frac{f^{\text{voltage}} \cdot v_j}{\tau_{l~j~i}^2} \ge \text{drop}_{l~i~j} - M(1 - d^{\text{master}}_{l~i~j})
    \end{align}

Where 

- :math:`\text{drop}_{l~i~j} = 2(r_l \cdot f^{\text{power}} \cdot p_{l~i~j} + x_l \cdot f^{\text{power}} \cdot q_{l~i~j}) - (r_l^2 + x_l^2) \cdot f^{\text{current}} \cdot i_{l~i~j}`.


- Tap-changing transformer effects are incorporated via :math:`\tau_{lij}`.
- If the line is not activate (:math:`d = 0`),  the constraint is deactivated using a Big-M relaxation.



2.4.7. Rotated Second Order Cone (SOC) Constraint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A **rotated second-order cone (SOC)** constraint ensures a convex relationship between the power flow and the current in each branch. It provides a safe approximation of the power flow physics in radial distribution networks and allows efficient optimization.

The SOC constraint is given by:

.. math::
    :label: soc-cone
    :nowrap:

    \begin{align*}
        (f^{\text{power}} \cdot p_{l~i~j})^2 + (f^{\text{power}} \cdot q_{l~i~j})^2 \le (\frac{f^{\text{voltage}} \cdot v_i}{\tau_{l~i~j}^2}) \cdot f^{\text{current}} \cdot i_{l~i~j}
    \end{align*}
    
    
with condition of:

- Deactivated for branches in switchable set :math:`S`.


2.4.8. Current Limit Constraint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This constrains allow soft violations in current when strict feasibility is not possible:


.. math::
    :label: current
    :nowrap:
    
    \begin{align}
        f^{\text{current}} \cdot i_{l~i~j} \le (i_{l}^{\max})^2 + i_{l~i~j}^{\text{slack}}
    \end{align}
    
- :math:`i_{l~i~j}^{\text{slack}}` is penalized in the objective if activated.
    
    
2.4.9. Voltage Bound Constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ensures voltage magnitudes remain within safety limits with slacks:

.. math::
    :label: Voltage Bounds 1
    :nowrap:
    
    \begin{align}
        f^{\text{voltage}} \cdot v_n \le (v_n^{\max})^2 + V^{+}_n
    \end{align}
    
.. math::
    :label: Voltage Bounds 2
    :nowrap:
    
    \begin{align}
        f^{\text{voltage}} \cdot v_n \ge (v_n^{\min})^2 - V^{-}_n
    \end{align}

- Slack variables allow soft violations when strict feasibility is not possible and are penalized in the objective function.





"""

import pyomo.environ as pyo


def slave_model_constraints(model: pyo.AbstractModel) -> pyo.AbstractModel:
    model.master_switch_status_propagation = pyo.Constraint(
        model.S, rule=master_switch_status_propagation_rule
    )
    # Distflow equations
    model.slack_voltage = pyo.Constraint(model.N, rule=slack_voltage_rule)
    model.node_active_power_balance = pyo.Constraint(
        model.N, rule=node_active_power_balance_rule
    )
    model.node_reactive_power_balance = pyo.Constraint(
        model.N, rule=node_reactive_power_balance_rule
    )
    model.voltage_drop_lower = pyo.Constraint(model.C, rule=voltage_drop_lower_rule)
    model.voltage_drop_upper = pyo.Constraint(model.C, rule=voltage_drop_upper_rule)

    model.current_rotated_cone = pyo.Constraint(model.C, rule=current_rotated_cone_rule)
    model.edge_active_power_balance = pyo.Constraint(
        model.L, rule=edge_active_power_balance_rule
    )
    model.edge_reactive_power_balance = pyo.Constraint(
        model.L, rule=edge_reactive_power_balance_rule
    )
    # Switch status constraints
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

    return model


def optimal_slave_model_constraints(model: pyo.AbstractModel) -> pyo.AbstractModel:
    model = slave_model_constraints(model)
    model.objective = pyo.Objective(rule=optimal_objective_rule, sense=pyo.minimize)
    # Physical limits
    model.current_limit = pyo.Constraint(model.C, rule=optimal_current_limit_rule)
    model.voltage_upper_limits = pyo.Constraint(
        model.N, rule=optimal_voltage_upper_limits_rule
    )
    model.voltage_lower_limits = pyo.Constraint(
        model.N, rule=optimal_voltage_lower_limits_rule
    )
    return model


def infeasible_slave_model_constraints(model: pyo.AbstractModel) -> pyo.AbstractModel:
    model = slave_model_constraints(model)
    model.objective = pyo.Objective(rule=infeasible_objective_rule, sense=pyo.minimize)
    # Physical limits
    model.current_limit = pyo.Constraint(model.C, rule=infeasible_current_limit_rule)
    model.voltage_upper_limits = pyo.Constraint(
        model.N, rule=infeasible_voltage_upper_limits_rule
    )
    model.voltage_lower_limits = pyo.Constraint(
        model.N, rule=infeasible_voltage_lower_limits_rule
    )
    return model


def optimal_objective_rule(m):

    return sum(m.r[l] * m.i_sq[l, i, j] for (l, i, j) in m.C)


def infeasible_objective_rule(m):
    v_penalty = sum(m.slack_v_pos[n] + m.slack_v_neg[n] for n in m.N)
    i_penalty = sum(m.slack_i_sq[l, i, j] for (l, i, j) in m.C)

    return v_penalty + i_penalty


def master_switch_status_propagation_rule(m, s):
    return m.delta[s] == m.master_delta[s]


# (1) Slack Bus: fix bus 0's voltage squared to 1.0.
# def slack_voltage_rule(m):
#     return m.v_sq[m.slack_node] == m.slack_node_v_sq
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


def current_balance_rule(m, l, i, j):
    return (
        m.i_sq[l, i, j] == m.i_sq[l, j, i]
    )  # Ensure current balance in both directions


# def current_rotated_cone_rule(m, l, i, j):
#     if l in m.S:
#         return pyo.Constraint.Skip
#     else:

#         lhs = (
#             (2 * m.p_flow[l, i, j]) ** 2
#             + (2 * m.q_flow[l, i, j]) ** 2
#             + (m.v_sq[i] / (m.n_transfo[l, i, j] ** 2) - m.i_sq[l, i, j]) ** 2
#         )
#         rhs = (m.v_sq[i] / (m.n_transfo[l, i, j] ** 2) + m.i_sq[l, i, j]) ** 2

#         return lhs <= rhs


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


def optimal_current_limit_rule(m, l, i, j):
    if l in m.S:
        return pyo.Constraint.Skip
    else:
        return m.i_sq[l, i, j] <= m.i_max[l] ** 2


def optimal_voltage_upper_limits_rule(m, n):
    return m.v_sq[n] <= m.v_max[n] ** 2


def optimal_voltage_lower_limits_rule(m, n):
    return m.v_sq[n] >= m.v_min[n] ** 2


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
