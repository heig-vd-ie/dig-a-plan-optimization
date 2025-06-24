r"""
1. Initialization
~~~~~~~~~~~~~~~~~~~~~

The **slave model** solves a **DistFlow optimization problem** for a fixed network topology (as determined by the master model).

The network topology is fixed by the master through parameters :math:`d^{master}_{l~i~j}` in slave model decomposition:


- In the **master model**, :math:`d_{l~i~j}` is a binary variable that selects whether candidate branch :math:`(l~i~j)` is part of the network.
- In the **slave model**, this value is passed as a fixed parameter :math:`d^{master}_{l~i~j}`.
- When :math:`d_{l~i~j} = 0`, the corresponding branch is disabled: power flows are zero, voltage constraints are relaxed, and SOC constraints are deactivated.
- This variable enables the decomposition of the overall problem into a master (topology planning) and a slave (power flow evaluation) problem.




2. Objective: Minimize losses with penalties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The optimization aims to minimize total resistive losses while softly enforcing operational constraints on voltage and current magnitudes.

.. math::
    :label: distflow-objective
    :nowrap:

    \begin{align}
        \text{Objective} =\ 
        \sum_{l~i~j} r_l \cdot i_{l~i~j}\ 
        + \lambda_v \cdot \sum_{n} V_n^{\text{slack}}\
        + \lambda_i \cdot \sum_{l~i~j} i_{l~i~j}^{\text{slack}}
    \end{align}

Where:

- The first term, :math:`\sum_{l~i~j} r_l \cdot i_{l~i~j}`, represents the total Joule heating losses in the network, i.e., :math:`P_{\text{loss}} = R \cdot I^2`.
- The second and third terms penalize constraint violations using slack variables that allow small violations of voltage and current limits.

Slack variables:

- :math:`V_n^{\text{slack}}`: allows soft violation of voltage bounds at node :math:`n`.
- :math:`i_{l~i~j}^{\text{slack}}`: allows soft violation of current limits in branch :math:`(l~i~j)`.

Penalty weights:

- :math:`\lambda_v`: penalty coefficient for voltage violations.
- :math:`\lambda_i`: penalty coefficient for current violations.

These penalty terms ensure feasibility while strongly discouraging violations unless absolutely necessary.


3. Slack bus
~~~~~~~~~~~~~~~~~~

The slack bus (or reference bus) is used to set the overall voltage level of the network. This constraint fixes the squared voltage at the slack node to a predetermined value:

----------------------

.. automodule:: slave_model.variables
   :no-index:

4. Power update
~~~~~~~~~~~~~~~~~~

.. math::
    :label: distflow-power
    :nowrap:

    \begin{align}
        P^{dn}_{l~i~j} &= d^{master}_{l~i~j} \cdot \left( -p_{j}^{\text{node}}
        - \sum_{\substack{l'~i'~j' \\ j'=i,\, i' \ne j}} P^{up}_{l'~i'~j'} \right) \\[1ex]
        Q^{dn}_{l~i~j} &= d^{master}_{l~i~j} \cdot \left( -q_{j}^{\text{node}}
        - \sum_{\substack{l'~i'~j' \\ j'=i,\, i' \ne j}} \left( Q^{up}_{l'~i'~j'} + \frac{b(l')}{2} v_i \right) \right) \\[1ex]
        P^{up}_{l~i~j} &= d^{master}_{l~i~j} \cdot \left( r(l) \cdot i_{l~i~j} - P^{dn}_{l~i~j} \right) \\[1ex]
        Q^{up}_{l~i~j} &= d^{master}_{l~i~j} \cdot \left( x(l) \cdot i_{l~i~j} - Q^{dn}_{l~i~j} \right)
    \end{align}


5. Voltage Drop Across Branch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::
    :label: voltage-drop
    :nowrap:
    
    \begin{align}

    \Delta v_{l~i~j} = -2 \cdot \left( r(l) \cdot P^{up}_{l~i~j} + x(l) \cdot Q^{up}_{l~i~j} \right)
    + \left( r(l)^2 + x(l)^2 \right) \cdot i_{l~i~j}
    
    \end{align}

Voltage drop constraints are enforced using transformer tap ratios and big-M logic:

.. math:: 
    :label: voltage-drop1
    :nowrap:
    
    \begin{align}

    \frac{v_i}{\tau_{l~i~j}} - \frac{v_j}{\tau_{l~i~j}} - \Delta v_{l~i~j} \in [-M(1-d^{master}_{l~i~j} ), M(1-d^{master}_{l~i~j} )]
    
    \end{align}
    


6. Rotated Second Order Cone (SOC) Constraint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::
    :label: distflow-rotated-cone-norm
    :nowrap:

    \begin{align*}
        & v_{j} + i_{l~i~j} \ge 
        \left\lVert
        \begin{pmatrix}
            2P^{\text{up}}_{l~i~j} \\
            2Q^{\text{up}}_{l~i~j} \\
            v_{j} - i_{l~i~j}
        \end{pmatrix}
        \right\rVert_2
    \end{align*}




7. Current (Flow) Bounds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::
    :label: current
    :nowrap:
    
    \begin{align}

    i_{l~i~j} \le i_{\max,l} + i_{l~i~j}^{\text{slack}}
    
    \end{align}
    
    
8. Voltage Bounds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each node voltage squared must remain within limits:

.. math::
    :label: Voltage Bounds 1
    :nowrap:
    
    \begin{align}
    v_n \le v_{\text{max},n} + V_n^{\text{slack}}
    \end{align}
.. math::
    :label: Voltage Bounds 2
    :nowrap:
    
    \begin{align}
    v_n \ge v_{\text{min},n} - V_n^{\text{slack}}
    \end{align}






"""
import pyomo.environ as pyo
from pyomo.environ import value

def slave_model_constraints(model: pyo.AbstractModel) -> pyo.AbstractModel:
    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)
    model.slack_voltage = pyo.Constraint(model.N, rule=slack_voltage_rule)
    
    model.node_active_power_balance = pyo.Constraint(model.LC, rule=node_active_power_balance_rule)
    model.node_reactive_power_balance = pyo.Constraint(model.LC, rule=node_reactive_power_balance_rule)
    model.voltage_drop_lower = pyo.Constraint(model.LC, rule=voltage_drop_lower_rule)
    model.voltage_drop_upper = pyo.Constraint(model.LC, rule=voltage_drop_upper_rule)
    model.current_rotated_cone = pyo.Constraint(model.LC, rule=current_rotated_cone_rule)
    model.current_limit = pyo.Constraint(model.LC, rule=infeasible_current_limit_rule)
    model.voltage_upper_limits = pyo.Constraint(model.N, rule=infeasible_voltage_upper_limits_rule)
    model.voltage_lower_limits = pyo.Constraint(model.N, rule=infeasible_voltage_lower_limits_rule)
    return model


def slave_model_constraints_2(model: pyo.AbstractModel) -> pyo.AbstractModel:
    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)
    model.slack_voltage = pyo.Constraint(model.N, rule=slack_voltage_rule)
    
    model.node_active_power_balance = pyo.Constraint(model.LC, rule=node_active_power_balance_rule)
    model.node_reactive_power_balance = pyo.Constraint(model.LC, rule=node_reactive_power_balance_rule)
    model.voltage_drop_lower = pyo.Constraint(model.LC, rule=voltage_drop_lower_rule)
    model.voltage_drop_upper = pyo.Constraint(model.LC, rule=voltage_drop_upper_rule)
    model.current_rotated_cone = pyo.Constraint(model.LC, rule=current_rotated_cone_rule_2)
    model.current_limit = pyo.Constraint(model.LC, rule=infeasible_current_limit_rule)
    model.voltage_upper_limits = pyo.Constraint(model.N, rule=infeasible_voltage_upper_limits_rule)
    model.voltage_lower_limits = pyo.Constraint(model.N, rule=infeasible_voltage_lower_limits_rule)
    return model



def test_slave_model_constraints(model: pyo.AbstractModel) -> pyo.AbstractModel:
    model.objective = pyo.Objective(rule=test_objective_rule, sense=pyo.minimize)
    model.slack_voltage = pyo.Constraint(model.N, rule=slack_voltage_rule)
    
    model.node_active_power_balance = pyo.Constraint(model.LC, rule=node_active_power_balance_rule)
    model.node_reactive_power_balance = pyo.Constraint(model.LC, rule=node_reactive_power_balance_rule)
    model.voltage_drop_lower = pyo.Constraint(model.LC, rule=voltage_drop_lower_rule)
    model.voltage_drop_upper = pyo.Constraint(model.LC, rule=voltage_drop_upper_rule)
    model.current_rotated_cone = pyo.Constraint(model.LC, rule=current_rotated_cone_rule)
    # model.current_limit = pyo.Constraint(model.LC, rule=feasible_current_limit_rule)
    # model.voltage_upper_limits = pyo.Constraint(model.N, rule=feasible_voltage_upper_limits_rule)
    # model.voltage_lower_limits = pyo.Constraint(model.N, rule=feasible_voltage_lower_limits_rule)

    return model

# def infeasible_slave_model_constraints(model: pyo.AbstractModel) -> pyo.AbstractModel:
#     model.objective = pyo.Objective(rule=infeasible_objective_rule, sense=pyo.minimize)
#     model.slack_voltage = pyo.Constraint(model.N, rule=slack_voltage_rule)
#     model.node_active_power_balance = pyo.Constraint(model.LC, rule=node_active_power_balance_rule)
#     model.node_reactive_power_balance = pyo.Constraint(model.LC, rule=node_reactive_power_balance_rule)
#     model.voltage_drop_lower = pyo.Constraint(model.LC, rule=voltage_drop_lower_rule)
#     model.voltage_drop_upper = pyo.Constraint(model.LC, rule=voltage_drop_upper_rule)
#     model.current_rotated_cone = pyo.Constraint(model.LC, rule=current_rotated_cone_rule)
#     model.current_limit = pyo.Constraint(model.LC, rule=infeasible_current_limit_rule)
#     model.voltage_upper_limits = pyo.Constraint(model.N, rule=infeasible_voltage_upper_limits_rule)
#     model.voltage_lower_limits = pyo.Constraint(model.N, rule=infeasible_voltage_lower_limits_rule)

#     return model


def objective_rule(m):
    line_losses = sum(m.r[l] * m.i_sq[l, i, j] for (l, i, j) in m.LC)
    # switch_losses = sum(10e-4 * m.i_sq[l, i, j] for (l, i, j) in m.LC if l in m.S)
    v_penalty = sum(m.slack_v_pos[n]  + m.slack_v_neg[n]  for n in m.N)
    i_penalty = sum(m.slack_i_sq[l, i, j] for (l, i, j) in m.LC)
    return line_losses  + m.penalty_cost *(v_penalty + i_penalty)

def test_objective_rule(m):
    edge_losses = sum(m.r[l] * m.i_sq[l, i, j] for (l, i, j) in m.LC)
    
    return edge_losses 

def infeasible_objective_rule(m):
    v_penalty = sum(m.slack_v_pos[n]  + m.slack_v_neg[n]  for n in m.N)
    i_penalty = sum(m.slack_i_sq[l, i, j] for (l, i, j) in m.LC)

    return v_penalty + i_penalty


# (1) Slack Bus: fix bus 0's voltage squared to 1.0.
# def slack_voltage_rule(m):
#     return m.v_sq[m.slack_node] == m.slack_node_v_sq
def slack_voltage_rule(m, n):
    if n == pyo.value(m.slack_node):
        return m.v_sq[n] == m.slack_node_v_sq
    return pyo.Constraint.Skip

# (2) Node Power Balance (Real) for candidate (l,i,j).
    # For candidate (l, i, j), j is the downstream bus.
    
def node_active_power_balance_rule(m, l, i, j):
    downstream_power_flow = sum(
        m.r[l_] * m.i_sq[l_, i_, j_] - m.p_flow[l_, i_, j_] for (l_, i_, j_) in m.LC if (j_ == i) and (i_ != j)
    )
    return m.p_flow[l, i, j] + m.master_d[l, i, j] * (m.p_node[i] + downstream_power_flow) == 0

def neg_node_active_power_balance_rule(m, l, i, j):
    downstream_power_flow = sum(
        m.r[l_] * m.i_sq[l_, i_, j_] - m.p_flow[l_, i_, j_] for (l_, i_, j_) in m.LC if (j_ == i) and (i_ != j)
    )
    return m.p_flow[l, i, j] + m.master_d[l, i, j] * (m.p_node[i] + downstream_power_flow) <= 0

def pos_node_active_power_balance_rule(m, l, i, j):
    downstream_power_flow = sum(
        m.r[l_] * m.i_sq[l_, i_, j_] - m.p_flow[l_, i_, j_] for (l_, i_, j_) in m.LC if (j_ == i) and (i_ != j)
    )
    return m.p_flow[l, i, j] + m.master_d[l, i, j] * (m.p_node[i] + downstream_power_flow) >= 0


# (3) Node Power Balance (Reactive) for candidate (l,i,j).
def node_reactive_power_balance_rule(m, l, i, j):
    downstream_power_flow = sum(
        m.x[l_] * m.i_sq[l_, i_, j_] - m.q_flow[l_, i_, j_] for (l_, i_, j_) in m.LC if (j_ == i) and (i_ != j)
    )
    transversal_power = sum(
        - m.b[l_]/2 * m.v_sq[i] for (l_, i_, _) in m.LC if (i_ == i)
    )
    return m.q_flow[l, i, j] + m.master_d[l, i, j] * (m.q_node[i] + downstream_power_flow + transversal_power) == 0

def neg_node_reactive_power_balance_rule(m, l, i, j):
    downstream_power_flow = sum(
        m.x[l_] * m.i_sq[l_, i_, j_] - m.q_flow[l_, i_, j_] for (l_, i_, j_) in m.LC if (j_ == i) and (i_ != j)
    )
    transversal_power = sum(
        - m.b[l_]/2 * m.v_sq[i] for (l_, i_, _) in m.LC if (i_ == i)
    )
    return m.q_flow[l, i, j] + m.master_d[l, i, j] * (m.q_node[i] + downstream_power_flow + transversal_power) <= 0

def pos_node_reactive_power_balance_rule(m, l, i, j):
    downstream_power_flow = sum(
        m.x[l_] * m.i_sq[l_, i_, j_] - m.q_flow[l_, i_, j_] for (l_, i_, j_) in m.LC if (j_ == i) and (i_ != j)
    )
    transversal_power = sum(
        - m.b[l_]/2 * m.v_sq[i] for (l_, i_, _) in m.LC if (i_ == i)
    )
    return m.q_flow[l, i, j] + m.master_d[l, i, j] * (m.q_node[i] + downstream_power_flow + transversal_power) >= 0

    
# (4) Voltage Drop along Branch for candidate (l,i,j).
# Let expr = v_sq[i] - 2*(r[l]*p_z_up(l,i,j) + x[l]*q_z_up(l,i,j)) + (r[l]^2+x[l]^2)*f_c(l,i,j).
# We then enforce two separate inequalities:
def voltage_drop_lower_rule(m, l, i, j):   
    dv =  2 * (m.r[l] * m.p_flow[l, i, j] + m.x[l]*m.q_flow[l, i, j]) - (m.r[l]**2 + m.x[l]**2) * m.i_sq[l, i, j]
    
    return  m.v_sq[i] / (m.n_transfo[l, i, j] ** 2) - m.v_sq[j] / (m.n_transfo[l, j, i] ** 2)  - dv >= - m.big_m*(1 - m.master_d[l, i, j])

def voltage_drop_upper_rule(m, l, i, j):
    dv =  2 * (m.r[l] * m.p_flow[l, i, j] + m.x[l]*m.q_flow[l, i, j]) - (m.r[l]**2 + m.x[l]**2) * m.i_sq[l, i, j]
    
    return  m.v_sq[i] / (m.n_transfo[l, i, j] ** 2) - m.v_sq[j] / (m.n_transfo[l, j, i] ** 2)  - dv <= m.big_m*(1 - m.master_d[l, i, j])

# (5) Rotated Cone (SOC) Current Constraint for candidate (l,i,j):
# Enforce: ||[2*p_z_up, 2*q_z_up, v_sq[i]-f(l,i,j)]||_2 <= v_sq[i]+f(l,i,j)
# In squared form: (2*p_z_up)^2 + (2*q_z_up)^2 + (v_sq[i] - f)^2 <= (v_sq[i] + f)^2.
def current_rotated_cone_rule_2(m, l, i, j):
    if l in m.S:
        return m.i_sq[l, i, j] == 0
    else:
        if value(m.master_d[l,i,j]) == 1:       
            lhs = (2*m.p_flow[l, i, j])**2 + (2*m.q_flow[l, i, j])**2 + (m.v_sq[i]/ (m.n_transfo[l, i, j] ** 2) - m.i_sq[l, i, j])**2
            rhs = (m.v_sq[i]/ (m.n_transfo[l, i, j] ** 2) + m.i_sq[l, i, j])**2
            
            return lhs <= rhs
        else:
            return m.i_sq[l, i, j] == 0

# def current_rotated_cone_rule(m, l, i, j):
#     if l in m.S:
#         return m.i_sq[l, i, j] == 0
#     else:

#         lhs = (2*m.p_flow[l, i, j])**2 + (2*m.q_flow[l, i, j])**2 + (m.v_sq[i]/ (m.n_transfo[l, i, j] ** 2) - m.i_sq[l, i, j])**2
#         rhs = (m.v_sq[i]/ (m.n_transfo[l, i, j] ** 2) + m.i_sq[l, i, j])**2


#         return lhs <= rhs

def current_rotated_cone_rule(m, l, i, j):
    if l in m.S:
        return m.i_sq[l, i, j] == 0
    else:

        lhs = m.p_flow[l, i, j]**2 + m.q_flow[l, i, j]**2
        rhs = (m.v_sq[i]/ (m.n_transfo[l, i, j] ** 2) * m.i_sq[l, i, j])

        
        return lhs <= rhs


# def current_rotated_cone_rule(m, l, i, j):
#     if l in m.S:
#         return m.i_sq[l, i, j] == 0
#     else:

#         lhs = m.p_flow[l, i, j]**2 + m.q_flow[l, i, j]**2
#         rhs = (m.v_sq[i]/ (m.n_transfo[l, i, j] ** 2) * m.i_sq[l, i, j])

        
#         return lhs <= rhs


####################################################################


# (6) Flow Bounds for candidate (l,i,j):
def feasible_current_limit_rule(m, l, i, j):
    return m.i_sq[l, i, j] <= m.i_max[l]**2
# (7) Voltage Limits: enforce v_sq[i] in [vmin^2, vmax^2].
def feasible_voltage_upper_limits_rule(m, n):
    return m.v_sq[n] <= m.v_max[n]**2 

def feasible_voltage_lower_limits_rule(m, n):
    return m.v_sq[n] >= m.v_min[n]**2

# (6) Flow Bounds for candidate (l,i,j):
def infeasible_current_limit_rule(m, l, i, j):
    return m.i_sq[l, i, j] <= m.i_max[l]**2 + m.slack_i_sq[l, i, j]
# (7) Voltage Limits: enforce v_sq[i] in [vmin^2, vmax^2].
def infeasible_voltage_upper_limits_rule(m, n):
    return m.v_sq[n] <= m.v_max[n]**2 + m.slack_v_pos[n]

def infeasible_voltage_lower_limits_rule(m, n):
    return m.v_sq[n] >= m.v_min[n]**2 - m.slack_v_neg[n]


