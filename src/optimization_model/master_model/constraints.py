r"""
1. Initialization
~~~~~~~~~~~~~~~~~~~~~

This section describes the optimization model used in the **master problem**, which selects the network topology and flow variables. 
The objective is to minimize resistive losses and a Benders auxiliary variable.

2. Objective Function
~~~~~~~~~~~~~~~~~~~~~~

.. math::
    :label: master-objective
    :nowrap:

    \begin{align}
        \min \sum_{l~i~j} r_l \cdot \left( p_{l~i~j}^2 + q_{l~i~j}^2 \right) + \Theta
    \end{align}

- The first term represents approximate losses across all branches.

- :math:`\Theta` is an auxiliary variable for Benders decomposition (used to include slave model costs).

3. Orientation Constraint
~~~~~~~~~~~~~~~~~~~~~~~~~

For each branch :math:`l`, the master model ensures that:

- If :math:`l` is switchable, it is active only when one of its associated candidate connections is selected.

- If :math:`l` is non-switchable, one candidate connection must be active.

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

4. Power Flow
~~~~~~~~~~~~~~~

The real and reactive power flows are restricted using Big-M logic:

.. math::
    :label: master-flow-limits
    :nowrap:

    \begin{align}
        -M \cdot d_{l~i~j} &\le p_{l~i~j} \le M \cdot d_{l~i~j} \\
        -M \cdot d_{l~i~j} &\le q_{l~i~j} \le M \cdot d_{l~i~j}
    \end{align}

These bounds ensure that when :math:`d_{l~i~j} = 0`, the flow is forced to zero.

    


5. Power Balance Constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For each bus :math:`n \ne \text{slack}`:

.. math::
    :label: master-balance
    :nowrap:

    \begin{align}
        \sum_{l~i~j:\, j=n} p_{l~i~j} - \sum_{l~i~j:\, i=n} p_{l~i~j} &= p_n^{\text{node}} \\
        \sum_{l~i~j:\, j=n} q_{l~i~j} - \sum_{l~i~j:\, i=n} q_{l~i~j} &= q_n^{\text{node}}
    \end{align}




6. Radiality Constraint
~~~~~~~~~~~~~~~~~~~~~~~~
Each non-slack node must have **exactly one** incoming active branch. For the slack node, this value is zero:

.. math::
    :label: master-radiality
    :nowrap:

    \begin{align}
        \sum_{l~i~j:\, j = n} d_{l~i~j} =
        \begin{cases}
            0 & \text{if } n = \text{slack node} \\
            1 & \text{otherwise}
        \end{cases}
    \end{align}









"""


# constraints.py
import pyomo.environ as pyo
from pyomo.environ import ConstraintList

def master_model_constraints(model: pyo.AbstractModel) -> pyo.AbstractModel:

    # model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)
    model.objective = pyo.Objective(rule=master_obj, sense=pyo.minimize)
    
    model.radiality = pyo.Constraint(model.N, rule=radiality_rule)
    model.orientation = pyo.Constraint(model.L, rule=orientation_rule)
    
    
    # model.ohmic_losses = pyo.Constraint(rule=ohmic_losses_rule)
    
    # model.flow_P_lower = pyo.Constraint(model.LC, rule=flow_P_lower_rule)
    # model.flow_P_upper = pyo.Constraint(model.LC, rule=flow_P_upper_rule)
    # model.flow_Q_lower = pyo.Constraint(model.LC, rule=flow_Q_lower_rule)
    # model.flow_Q_upper = pyo.Constraint(model.LC, rule=flow_Q_upper_rule)
    # model.power_balance_real = pyo.Constraint(model.N, rule=power_balance_real_rule)
    # model.power_balance_reactive = pyo.Constraint(model.N, rule=power_balance_reactive_rule)
    
    # model.slack_voltage = pyo.Constraint(model.N, rule=slack_voltage_rule)
    
    # model.volt_drop_lower = pyo.Constraint(model.LC, rule=volt_drop_lower_rule)
    # model.volt_drop_upper = pyo.Constraint(model.LC, rule=volt_drop_upper_rule)
    
    # model.voltage_upper_limits = pyo.Constraint(model.N, rule=voltage_upper_limits_rule)
    # model.voltage_lower_limits = pyo.Constraint(model.N, rule=voltage_lower_limits_rule)
    
    
    # cuts are generated on-the-fly, so no rules are necessary.
    model.infeasibility_cut = pyo.ConstraintList()
    model.optimality_cut = pyo.ConstraintList()
    
    return model

# Objective: approximate losses + Benders cuts
def master_obj(m):
    # v_penalty = sum(m.slack_v_pos[n]  + m.slack_v_neg[n]  for n in m.N)
    # return m.theta  + m.penalty_cost*v_penalty + m.losses
    return m.theta 

def ohmic_losses_rule(m):
    # Objective function to minimize resistive losses.
    return sum(m.r[l] * (m.p_flow[l, i, j]**2 + m.q_flow[l, i, j]**2) for (l, i, j) in m.LC) == m.losses

# def theta_initialization_rule(m):
#     # Initialize theta to a large value to ensure it is non-negative.
#     return m.theta == 0  # or any sufficiently large value 

def slack_voltage_rule(m, n):
    if n == pyo.value(m.slack_node):
        return m.v_sq[n] == m.slack_node_v_sq
    else:
        return pyo.Constraint.Skip 

# Orientation constraint.
def orientation_rule(m, l):

    if l in m.S:
        return sum(m.d[l_, i, j] for (l_, i, j) in m.LC if l_ == l) == m.delta[l]
    else:
        return sum(m.d[l_, i, j] for (l_, i, j) in m.LC if l_ == l) == 1

# Big-M constraints for real power flows.
def flow_P_lower_rule(m, l, i, j):
    return m.p_flow[l, i, j] >= -m.big_m * m.d[l, i, j]

def flow_P_upper_rule(m, l, i, j):
    return m.p_flow[l, i, j] <= m.big_m * m.d[l, i, j]

# Big-M constraints for reactive power flows.
def flow_Q_lower_rule(m, l, i, j):
    return m.q_flow[l, i, j] >= -m.big_m * m.d[l, i, j]

def flow_Q_upper_rule(m, l, i, j):
    return m.q_flow[l, i, j] <= m.big_m * m.d[l, i, j]

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
    
# voltage dropped lower bound 
def volt_drop_lower_rule(m, l, i, j):
    dv = (m.r[l]*m.p_flow[l,i,j] + m.x[l]*m.q_flow[l,i,j]) 
    return  m.v_sq[i] / (m.n_transfo[l, i, j] ** 2) - m.v_sq[j] / (m.n_transfo[l, j, i] ** 2)  - dv >= - m.big_m*(1 - m.d[l, i, j])

        
# Voltage dropped upper bound 

def volt_drop_upper_rule(m, l, i, j):
    dv = (m.r[l]*m.p_flow[l,i,j] + m.x[l]*m.q_flow[l,i,j]) 
    return m.v_sq[i] / (m.n_transfo[l, i, j] ** 2) - m.v_sq[j] / (m.n_transfo[l, j, i] ** 2)  - dv <= m.big_m*(1 - m.d[l, i, j])
    
# Voltage Limits: enforce v_sq[i] in [vmin^2, vmax^2].
def voltage_upper_limits_rule(m, n):
    return m.v_sq[n] <= m.v_max[n]**2 + m.slack_v_pos[n]

def voltage_lower_limits_rule(m, n):
    return m.v_sq[n] >= m.v_min[n]**2 - m.slack_v_neg[n]

        
    
