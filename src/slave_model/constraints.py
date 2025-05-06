r"""
1. Initialization
~~~~~~~~~~~~~~~~~~~~~

.. math::
    :label: distflow-initialization
    :nowrap:
    
    \begin{align} 
        v_{i} &= \frac{v_{0}}{\tau_{0\, \to\, i}} \\
        P^{up}_{z} &= \displaystyle\sum_{\Large{d\, \in\, N_i}} p^{\text{node}}_{d}\\
        Q^{up}_{z} &= \displaystyle\sum_{\Large{d\, \in\, N_i}} q^{\text{node}}_{d}
    \end{align}

2. Power update
~~~~~~~~~~~~~~~~~~

.. math::
    :label: distflow-power
    :nowrap:
    
    \begin{align}
        P_{z}^{\text{dn}}(l,i,j) &= -\,p_{\text{node}}(j)
        - \sum_{\substack{(l',i',j') \\ i' = j,\; j' \neq i}} P_{z}^{\text{up}}(l',i',j'), \label{eq:active_balance} \\[1ex]
        Q_{z}^{\text{dn}}(l,i,j) &= -\,q_{\text{node}}(j)
        - \sum_{\substack{(l',i',j') \\ i' = j,\; j' \neq i}}
        \Bigl( Q_{z}^{\text{up}}(l',i',j') + b(l')\,v_{j}^{2} \Bigr), \label{eq:reactive_balance} \\[1ex]
        P_{z}^{\text{up}}(l,i,j) &= r(l)\,i^{2} - P_{z}^{\text{dn}}(l,i,j), \label{eq:active_flow} \\[1ex]
        Q_{z}^{\text{up}}(l,i,j) &= x(l)\,i^{2} - Q_{z}^{\text{dn}}(l,i,j). \label{eq:reactive_flow}
    \end{align}


3. Voltage update
~~~~~~~~~~~~~~~~~~~~~

.. math::
    :label: distflow-voltage
    :nowrap:
    
    \begin{align} 
        \Delta v_{i} & = -2\left(r(l)\,P_{z}^{\text{up}}(l,i,j) + x(l)\,Q_{z}^{\text{up}}(l,i,j)\right)
        + \left(r(l)^{2} + x(l)^{2}\right)i^{2}

    \end{align}

5. Rotated Second Order Cone (SOC) Constraint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::
    :label: distflow-initialization-matrix
    :nowrap:
    
    \begin{align} 
        \left(2\,P_{z}^{\text{up}}(l,i,j)\right)^{2} + \left(2\,Q_{z}^{\text{up}}(l,i,j)\right)^{2}
        + \Bigl(v_{i}^{2}-i^{2}\Bigr)^{2} \le \Bigl(v_{i}^{2}+i^{2}\Bigr)^{2}

        
    \end{align}


6. Objective: Minimize losses with penalties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Resistive losses: In each branch (l,i,j), the real-power loss is:


.. math::
    :label: distflow-voltage-matrix
    :nowrap:
    
    \begin{align}
        \text{Objective} = \sum_{(l,i,j)} r_l \cdot i^2_{l,ij} + v\_penalty + i\_penalty
    \end{align}

Where:

-    The first term, :math:`\sum r_l \cdot i^2_{l,ij}`, represents the total line (edge) losses, corresponding to Joule heating losses :math: `(P_{\text{loss}} = R \cdot I^2)` across all branches.
-   :math: The second and third terms, :math:`v\_penalty` and :math: `i\_penalty`, are regularization terms that encourage voltages and currents to remain within their prescribed limits.

Penalties terms
---------------

They are penalty terms that "soften" the hard limits on voltage bounds and current limits in the model.

In theory:

- Voltage at each bus must stay between :math:`V_{\text{min}}` and :math:`V_{\text{max}}`.
- Current in each branch must stay below :math:`I_{\text{max}}`.

Thus, "slack variables" are introduced:

- :math:`\texttt{slack\_v\_sq[n]}` allows small violations of the voltage constraints.
- :math:`\texttt{slack\_i\_sq[l,i,j]}` allows small violations of the current constraints.

Violating constraints must nevertheless be penalized heavily; otherwise, the solver could abuse these slacks to find easier but unrealistic solutions.




Let's consider nodes :math:`1`, :math:`2` and :math:`3` sequencly connected to the slack node :math:`0`. 
The voltage level at node :math:`3`, calculated with reference to the slack node voltage, is given by the following expression:

.. math::
    :label: voltage_calculation_explanation
    :nowrap:
    
    \begin{align} 
        \large{
            \begin{cases}
            v_{1} = \frac{v_{0}}{N_{1}^2} + dv_{1} \\
            v_{2} = \frac{v_{1}}{N_{2}^2} + dv_{2} \\
            v_{3} = \frac{v_{2}}{N_{3}^2} + dv_{3}
            \end{cases}
        } \Rightarrow v_{3} = \frac{v_{0}}{N_{1}^2 \cdot N_{2}^2 \cdot N_{3}^2} +  \frac{dv_{1}}{N_{2}^2 \cdot N_{3}^2} + \frac{dv_{2}}{N_{3}^2} + dv_{3} \\
    \end{align}

Therefore, by recursive substitution, we can express the general voltage level for any node :math:`i` as:

.. math::
    :label: voltage-calculation-final
    :nowrap:
    
    \begin{align} 
        v_{i} = \frac{v_{0}}{\tau_{0\, \to\, i}} + \displaystyle\sum_{\Large{p \,\in\, P_0^i}} \frac{dv_{p}}{\tau_{p\, \to\, i}}
    \end{align}

Where :math:`\tau_{p\, \to\, i}` is given by the following expression:

.. math::
    :label: tau-calculation
    :nowrap:
    
    \begin{align} 
        \tau_{p\, \to\, i} = \displaystyle\prod_{\Large{q \,\in\, P_p^i}} N_{q}^{2}
    \end{align}

"""

import pyomo.environ as pyo

def slave_model_constraints(model: pyo.AbstractModel) -> pyo.AbstractModel:
    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)
    model.slack_voltage = pyo.Constraint(rule=slack_voltage_rule)
    model.node_active_power_balance = pyo.Constraint(model.LC, rule=node_active_power_balance_rule)
    model.node_reactive_power_balance = pyo.Constraint(model.LC, rule=node_reactive_power_balance_rule)
    model.active_power_flow = pyo.Constraint(model.LC, rule=active_power_flow_rule)
    model.reactive_power_flow = pyo.Constraint(model.LC, rule=reactive_power_flow_rule)
    model.voltage_drop_lower = pyo.Constraint(model.LC, rule=voltage_drop_lower_rule)
    model.voltage_drop_upper = pyo.Constraint(model.LC, rule=voltage_drop_upper_rule)
    model.current_rotated_cone = pyo.Constraint(model.LC, rule=current_rotated_cone_rule)
    model.current_limit = pyo.Constraint(model.LC, rule=current_limit_rule)
    model.voltage_upper_limits = pyo.Constraint(model.N, rule=voltage_upper_limits_rule)
    model.voltage_lower_limits = pyo.Constraint(model.N, rule=voltage_lower_limits_rule)
    return model
    

def objective_rule(m):
    edge_losses = sum(m.r[l] * m.i_sq[l, i, j] for (l, i, j) in m.LC)
    v_penalty = m.v_penalty_cost * sum(m.slack_v_sq[n] for n in m.N)
    i_penalty = m.v_penalty_cost * sum(m.slack_i_sq[l, i, j] for (l, i, j) in m.LC)
    return edge_losses + v_penalty + i_penalty

# (1) Slack Bus: fix bus 0's voltage squared to 1.0.
def slack_voltage_rule(m):
    return m.v_sq[m.slack_node] == m.slack_node_v_sq
    # return m.v_sq[m.slack_node] == m.slack_voltage

# (32 Node Power Balance (Real) for candidate (l,i,j).
    # For candidate (l, i, j), j is the downstream bus.
def node_active_power_balance_rule(m, l, i, j):
    downstream_power_flow = sum(
        m.p_z_up[l_, i_, j_] for (l_, i_, j_) in m.LC if (j_ == i) and (i_ != j)
    )
    return m.p_z_dn[l, i, j] ==m.master_d[l, i, j] * (- m.p_node[i] - downstream_power_flow)


# (2) Node Power Balance (Reactive) for candidate (l,i,j).
def node_reactive_power_balance_rule(m, l, i, j):
    downstream_power_flow = sum(
        m.q_z_up[l_, i_, j_] for (l_, i_, j_) in m.LC if (j_ == i) and (i_ != j)
    )
    transversal_power = sum(
        - m.b[l_]/2 * m.v_sq[i] for (l_, i_, _) in m.LC if (i_ == i)
    )
    return m.q_z_dn[l, i, j] == m.master_d[l, i, j] * (- m.q_node[i] - downstream_power_flow - transversal_power)

# (3) Upstream Flow Definitions for candidate (l,i,j):
def active_power_flow_rule(m, l, i, j):
    return m.p_z_up[l, i, j] == m.master_d[l, i, j] * (m.r[l] * m.i_sq[l, i, j] - m.p_z_dn[l, i, j])

def reactive_power_flow_rule(m, l, i, j):
    return m.q_z_up[l, i, j] == m.master_d[l, i, j] * (m.x[l] * m.i_sq[l, i, j] - m.q_z_dn[l, i, j])
    
# (4) Voltage Drop along Branch for candidate (l,i,j).
# Let expr = v_sq[i] - 2*(r[l]*p_z_up(l,i,j) + x[l]*q_z_up(l,i,j)) + (r[l]^2+x[l]^2)*f_c(l,i,j).
# We then enforce two separate inequalities:
def voltage_drop_lower_rule(m, l, i, j):
    dv = - 2*(m.r[l]*m.p_z_up[l, i, j] + m.x[l]*m.q_z_up[l, i, j]) + (m.r[l]**2 + m.x[l]**2)*m.i_sq[l, i, j]
    return  m.v_sq[i] / (m.n_transfo[l, i, j] ** 2) - m.v_sq[j] / (m.n_transfo[l, j, i] ** 2)  - dv >= - m.big_m*(1 - m.master_d[l, i, j])

def voltage_drop_upper_rule(m, l, i, j):
    dv = - 2*(m.r[l]*m.p_z_up[l, i, j] + m.x[l]*m.q_z_up[l, i, j]) + (m.r[l]**2 + m.x[l]**2)*m.i_sq[l, i, j]
    return  m.v_sq[i] / (m.n_transfo[l, i, j] ** 2) - m.v_sq[j] / (m.n_transfo[l, j, i] ** 2)  - dv <= m.big_m*(1 - m.master_d[l, i, j])

# (5) Rotated Cone (SOC) Current Constraint for candidate (l,i,j):
# Enforce: ||[2*p_z_up, 2*q_z_up, v_sq[i]-f(l,i,j)]||_2 <= v_sq[i]+f(l,i,j)
# In squared form: (2*p_z_up)^2 + (2*q_z_up)^2 + (v_sq[i] - f)^2 <= (v_sq[i] + f)^2.
def current_rotated_cone_rule(m, l, i, j):
    if l in m.S:
        return m.i_sq[l, i, j] == 0
    else:
        lhs = (2*m.p_z_up[l, i, j])**2 + (2*m.q_z_up[l, i, j])**2 + (m.v_sq[j]/ (m.n_transfo[l, j, i] ** 2) - m.i_sq[l, i, j])**2
        rhs = (m.v_sq[j]/ (m.n_transfo[l, j, i] ** 2) + m.i_sq[l, i, j])**2
        return m.master_d[l, i, j] * lhs <= rhs


# (6) Flow Bounds for candidate (l,i,j):
def current_limit_rule(m, l, i, j):
    return m.i_sq[l, i, j] <= m.i_max[l]**2 + m.slack_i_sq[l, i, j]

# (7) Voltage Limits: enforce v_sq[i] in [vmin^2, vmax^2].
def voltage_upper_limits_rule(m, n):
    return m.v_sq[n] <= m.v_max[n]**2 + m.slack_v_sq[n]

def voltage_lower_limits_rule(m, n):
    return m.v_sq[n] >= m.v_min[n]**2 - m.slack_v_sq[n]


