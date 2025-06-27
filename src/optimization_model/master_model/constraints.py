r"""
1.4.1. Initialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This section describes the optimization model used in the **master problem**, which selects the network topology and flow variables. 
The objective is to minimize resistive losses and a Benders auxiliary variable.

1.4.2. Objective Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::
    :label: master-objective
    :nowrap:

    \begin{align}
        \min \Theta
    \end{align}

- :math:`\Theta` is an auxiliary variable used in Benders decomposition to represent the lower bound on the total cost. Resistive losses are handled implicitly in the slave problem.

1.4.3. Slack Voltage Constraint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To anchor the voltage profile, the squared voltage magnitude at the slack node is fixed to a known value. This ensures the voltage reference point is defined and consistent across iterations:

.. math::
    :label: master-slack-voltage
    :nowrap:

    \begin{align}
        V_n = V_{\text{slack}} \quad \text{if } n = \text{slack}_{\text{node}}
    \end{align}

- This constraint is only applied at the slack node. For all other nodes, the condition is skipped.
- :math:`V_{\text{slack}}` is a model parameter defined externally and used to initialize or maintain voltage consistency.


1.4.4. Orientation Constraint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This constraint determines how the binary variables control branch status:

- If the branch :math:`l` is switchable (i.e., :math:`l \in S`), then its activation depends on the switch status :math:`\delta_l`.
- If the branch is not switchable, then exactly one candidate connection must be selected.

.. math::
    :label: master-orientation
    :nowrap:

    \begin{align}
        d_{l~i~j} + d_{l~j~i} =
        \begin{cases}
            \delta_l & \text{if } l \in S \\
            1 & \text{otherwise}
        \end{cases}
    \end{align}


1.4.5. Radiality Constraint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Each non-slack node must have exactly one incoming active branch, while the slack node must have zero:

.. math::
    :label: master-radiality
    :nowrap:

    \begin{align}
        \sum_{(l~i~j) \in LC: i = n} d_{l~i~j} =
        \begin{cases}
            0 & \text{if } n = \text{slack}_{\text{node}} \\
            1 & \text{otherwise}
        \end{cases}
    \end{align}


1.4.6. Power Flow Constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These constraints enforce upper and lower bounds on real power flows using Big-M formulation:

.. math::
    :label: master-flow-limits
    :nowrap:

    \begin{align}
        -M \cdot d_{l~i~j} &\le p_{l~i~j} \le M \cdot d_{l~i~j}
    \end{align}

This ensures that when a line is inactive (:math:`d_{l~i~j}` = 0), its power flow is also zero.


1.4.7. Power Balance Constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For each bus :math:`n \ne \text{slack}`:

.. math::
    :label: master-balance
    :nowrap:

    \begin{align}
        \sum_{(l~i~j) \in LC: j=n} p_{l~i~j} - \sum_{(l~i~j) \in LC: i=n} p_{l~i~j} = -1
    \end{align}

This reflects a constant net demand of 1 unit at each non-slack bus, while the slack bus provides the balancing power.




1.4.8. Bender cuts Constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Two constraint lists are reserved for dynamic Benders cuts:

- infeasibility_cut: captures configurations that violate feasibility in the slave problem.

- optimality_cut: bounds the objective value using dual information from the slave.

These are populated during the iterative Benders loop.




"""


# constraints.py
import pyomo.environ as pyo
from pyomo.environ import ConstraintList


def master_model_constraints(model: pyo.AbstractModel) -> pyo.AbstractModel:
    
    model.objective = pyo.Objective(rule=master_obj, sense=pyo.minimize)
    
    model.radiality = pyo.Constraint(model.N, rule=radiality_rule)
    model.orientation = pyo.Constraint(model.L, rule=orientation_rule)

    
    model.flow_P_lower = pyo.Constraint(model.LC, rule=flow_P_lower_rule)
    model.flow_P_upper = pyo.Constraint(model.LC, rule=flow_P_upper_rule)

    model.power_balance_real = pyo.Constraint(model.N, rule=power_balance_real_rule)

    
    # cuts are generated on-the-fly, so no rules are necessary.
    model.infeasibility_cut = pyo.ConstraintList()
    model.optimality_cut = pyo.ConstraintList()
    
    return model

# Objective: approximate losses + Benders cuts
def master_obj(m):
    return m.theta 

# Radiality constraint: each non-slack bus must have one incoming candidate.
def radiality_rule(m, n):
    if n == m.slack_node:
        return sum(m.d[l, a, b] for (l, a, b) in m.LC if a == n) == 0
    else:
        return sum(m.d[l, a, b] for (l, a, b) in m.LC if a == n) == 1
    
# Orientation constraint.
def orientation_rule(m, l):

    if l in m.S:
        return sum(m.d[l_, i, j] for (l_, i, j) in m.LC if l_ == l) == m.delta[l]
    else:
        return sum(m.d[l_, i, j] for (l_, i, j) in m.LC if l_ == l) == 1

def slack_voltage_rule(m, n):
    if n == pyo.value(m.slack_node):
        return m.v_sq[n] == m.slack_node_v_sq
    else:
        return pyo.Constraint.Skip 

# Big-M constraints for real power flows.
def flow_P_lower_rule(m, l, i, j):
    return m.p_flow[l, i, j] >= -m.big_m * m.d[l, i, j]

def flow_P_upper_rule(m, l, i, j):
    return m.p_flow[l, i, j] <= m.big_m * m.d[l, i, j]

# Real power balance.
def power_balance_real_rule(m, n):
    if n == m.slack_node:
        return pyo.Constraint.Skip
    p_in = sum(m.p_flow[l, a, b] for (l, a, b) in m.LC if b == n)
    p_out = sum(m.p_flow[l, a, b] for (l, a, b) in m.LC if a == n)
    return p_in - p_out == -1

#     # Reactive power balance.
# def power_balance_reactive_rule(m, n):
#     if n == m.slack_node:
#         return pyo.Constraint.Skip
#     q_in = sum(m.q_flow[l, a, b] for (l, a, b) in m.LC if b == n)
#     q_out = sum(m.q_flow[l, a, b] for (l, a, b) in m.LC if a == n)
#     return q_in - q_out == m.q_node[n]

# # Big-M constraints for reactive power flows.
# def flow_Q_lower_rule(m, l, i, j):
#     return m.q_flow[l, i, j] >= -m.big_m * m.d[l, i, j]

# def flow_Q_upper_rule(m, l, i, j):
#     return m.q_flow[l, i, j] <= m.big_m * m.d[l, i, j]

# def ohmic_losses_rule(m):
#     # Objective function to minimize resistive losses.
#     return sum(m.r[l] * (m.p_flow[l, i, j]**2 + m.q_flow[l, i, j]**2) for (l, i, j) in m.LC) == m.losses

# def theta_initialization_rule(m):
#     # Initialize theta to a large value to ensure it is non-negative.
#     return m.theta == 0  # or any sufficiently large value 
        
    
