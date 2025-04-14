r"""
1. Initialization
~~~~~~~~~~~~~~~~~~~~~

.. math::
    :label: distflow-initialization
    :nowrap:
    
    \begin{align} 
        v_{i} &= \frac{v_{0}}{\tau_{0\, \to\, i}} \\
        \overline{S^{z}_{i}} &= \displaystyle\sum_{\Large{d\, \in\, D_i}} S^{\text{node}}_{d}
    \end{align}

2. Power update
~~~~~~~~~~~~~~~~~~

.. math::
    :label: distflow-power
    :nowrap:
    
    \begin{align} 
        \underline{S^{\text{Z}}_{i}} &= - S^{\text{node}}_{i} - v_{i} \cdot \underline{Y_{i}} - \displaystyle\sum_{\Large{c\, \in\, C_i}} \left(\overline{S^{\text{Z}}_{c}} + v_{i} \cdot \overline{Y_{c}} \right)\\
        i_{i} &= \frac{\vert \underline{S^{\text{Z}}_{i}}\vert^{2}}{v_{i}}\\
        \overline{S^{\text{Z}}_{i}} &= i_{i} \cdot Z_{i} - \underline{S^{\text{Z}}_{i}}
    \end{align}

The negative sign around :math:`\underline{S^{\text{Z}}_{i}}` comes from :ref:`sign convention<sign-convention>`

3. Voltage update
~~~~~~~~~~~~~~~~~~~~~

.. math::
    :label: distflow-voltage
    :nowrap:
    
    \begin{align} 
        dv_{i} &=  - 2 \cdot \Re \left(Z_{i}^{*} \cdot \overline{S^{\text{Z}}_{i}} \right) + \vert Z_{i} \vert^{2} \cdot i_{i} \\
        v_{i}^{new} &= \frac{v_{0}}{\tau_{0\, \to\, i}} + \displaystyle\sum_{\Large{p \,\in\, P_0^i}} \frac{dv_{p}}{\tau_{p\, \to\, i}}
    \end{align}

4. Convergence check
~~~~~~~~~~~~~~~~~~~~~

.. math::
    :label: Convergence-check
    :nowrap:

    \begin{align} 
        \max_{i\, \in \, I} \frac{\vert v_{i}  - v_{i}^{new}\vert}{v_{i}^{new}} \leq tol \quad \begin{cases}
            \text{TRUE} \Rightarrow \text{STOP} \\
            \text{FALSE} \Rightarrow \text{GO TO 2}
        \end{cases} \
    \end{align}

Algorithm in matrix form
-------------------------

1. Matrix initialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::
    :label: distflow-initialization-matrix
    :nowrap:
    
    \begin{align} 
        \mathbf{V} &= \mathbf{V_{IN}} \\
        \mathbf{\overline{S_{Z}}} &= \mathbf{D} \times \mathbf{S_{\text{node}}}
        
    \end{align}

2. Matrix power update
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::
    :label: distflow-power-matrix
    :nowrap:
    
    \begin{align} 
        \mathbf{\underline{S_{Z}}} &= - \mathbf{S_{node}} - \mathbf{V} \odot \mathbf{Y_{tot}} \\
        \mathbf{I_{Z}} &= \mathbf{\underline{S_{Z}}} \odot \mathbf{\underline{S_{Z}}}^{*} \oslash \mathbf{V} \\
        \mathbf{\overline{S_{Z}}} &= \mathbf{I_{Z}} \odot \mathbf{Z} - \mathbf{\underline{S_{Z}}}
    \end{align}

3. Matrix voltage update
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::
    :label: distflow-voltage-matrix
    :nowrap:
    
    \begin{align} 
        \mathbf{dV} &=  - 2 \cdot \Re \left(\mathbf{Z}^{*} \odot \mathbf{\overline{S_{Z}}} \right) + \mathbf{Z^2} \odot \mathbf{I_{Z}} \\
        \mathbf{V} &= \mathbf{V_{IN}} + \mathbf{D_{prop}} \times \mathbf{dV}
    \end{align}

Where:

-   :math:`\odot` is the Hadamard product (element wise multiplication)
-   :math:`\oslash` is the Hadamard division
-   :math:`\times` is the matrix multiplication

Voltage drop demonstration
------------------------------

The general voltage drop decomposition between two nodes is given by the following equation (for branch n always equals 1):

.. math::
    :label: voltage-drop-decomposition
    :nowrap:
    
    \begin{align} 
        \frac{V_{i}}{N_{j}} - V_{j} &= Z_{j} \cdot I_{j} \\
        v_{j} & = \vert \frac{V_{i}}{N_{j}} -Z_{j} \cdot I_{j} \vert^{2} \\ 
        v_{j} & = \frac{v_{i}}{N_{j}^2} - 2 \cdot \Re \left(\frac{V_{i}}{N_{j}} \cdot Z_{j}^{*} \cdot I_{j}^{*} \right) + \vert Z_{j} \cdot I_{j}\vert^{2}\\
        v_{j} & = \frac{v_{i}}{N_{j}^2}  - 2 \cdot \Re \left(Z_{j}^{*} \cdot \overline{S^{\text{Z}}_{j}} \right) + \vert Z_{j} \vert^{2} \cdot i_{j} \\
        v_{j} & = \frac{v_{i}}{N_{j}^2} + dv_{j} \\
    \end{align}  

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
    # model.linking_f = pyo.Constraint(model.LF, rule=linking_f_rule)
    model.slack_voltage = pyo.Constraint(rule=slack_voltage_rule)
    model.voltage_limits = pyo.Constraint(model.N, rule=voltage_limits_rule)
    model.node_active_power_balance = pyo.Constraint(model.LC, rule=node_active_power_balance_rule)
    model.node_reactive_power_balance = pyo.Constraint(model.LC, rule=node_reactive_power_balance_rule)
    model.active_power_flow = pyo.Constraint(model.LC, rule=active_power_flow_rule)
    model.reactive_power_flow = pyo.Constraint(model.LC, rule=reactive_power_flow_rule)
    model.voltage_drop_lower = pyo.Constraint(model.LC, rule=voltage_drop_lower_rule)
    model.voltage_drop_z_upper = pyo.Constraint(model.LC, rule=voltage_drop_z_upper_rule)
    model.current_rotated_cone = pyo.Constraint(model.LC, rule=current_rotated_cone_rule)
    model.flow_bounds_real = pyo.Constraint(model.LC, rule=current_limit_rule)
    return model
    

# (1) Slack Bus: fix bus 0's voltage squared to 1.0.
def slack_voltage_rule(m):
    return m.v_sq[m.slack_node] == m.slack_v_sq
    # return m.v_sq[m.slack_node] == m.slack_voltage

# (3) Node Power Balance (Real) for candidate (l,i,j).
    # For candidate (l, i, j), j is the downstream bus.  
def node_active_power_balance_rule(m, l, i, j):
    children_sum = sum(
        m.p_z_up[l_2, i_2, j_2]
        for (l_2, i_2, j_2) in m.LC if (i_2 == j) and (j_2 != i)
    )
    return m.p_z_dn[l, i, j] == m.master_d[l, i, j] * (- m.p_node[j] - children_sum)


# (4) Node Power Balance (Reactive) for candidate (l,i,j).
def node_reactive_power_balance_rule(m, l, i, j):
    children_sum = sum(
        m.q_z_up[l_2, i_2, j_2] + m.b[l_2] * m.v_sq[j]
        for (l_2, i_2, j_2) in m.LC if (i_2 == j) and (j_2 != i)
    )

    return m.q_z_dn[l, i, j] == m.master_d[l, i, j] * (- m.q_node[j] - children_sum)

# (5) Upstream Flow Definitions for candidate (l,i,j):
def active_power_flow_rule(m, l, i, j):
    return m.p_z_up[l, i, j] == m.master_d[l, i, j] * (m.r[l] * m.i_sq[l, i, j] - m.p_z_dn[l, i, j])

def reactive_power_flow_rule(m, l, i, j):
        return m.q_z_up[l, i, j] == m.master_d[l, i, j] * (m.x[l] * m.i_sq[l, i, j] - m.q_z_dn[l, i, j])
    
# (6) Voltage Drop along Branch for candidate (l,i,j).
# Let expr = v_sq[i] - 2*(r[l]*p_z_up(l,i,j) + x[l]*q_z_up(l,i,j)) + (r[l]^2+x[l]^2)*f_c(l,i,j).
# We then enforce two separate inequalities:
def voltage_drop_lower_rule(m, l, i, j):
    dv = - 2*(m.r[l]*m.p_z_up[l, i, j] + m.x[l]*m.q_z_up[l, i, j]) + (m.r[l]**2 + m.x[l]**2)*m.i_sq[l, i, j]
    return m.v_sq[j] - m.v_sq[i] - dv >= - m.M*(1 - m.master_d[l, i, j])

def voltage_drop_z_upper_rule(m, l, i, j):
    dv = - 2*(m.r[l]*m.p_z_up[l, i, j] + m.x[l]*m.q_z_up[l, i, j]) + (m.r[l]**2 + m.x[l]**2)*m.i_sq[l, i, j]
    return m.v_sq[j] - m.v_sq[i] - dv <= m.M*(1 - m.master_d[l, i, j])
# (7) Rotated Cone (SOC) Current Constraint for candidate (l,i,j):
# Enforce: ||[2*p_z_up, 2*q_z_up, v_sq[i]-f(l,i,j)]||_2 <= v_sq[i]+f(l,i,j)
# In squared form: (2*p_z_up)^2 + (2*q_z_up)^2 + (v_sq[i] - f)^2 <= (v_sq[i] + f)^2.
def current_rotated_cone_rule(m, l, i, j):
    lhs = (2*m.p_z_up[l, i, j])**2 + (2*m.q_z_up[l, i, j])**2 + (m.v_sq[i] - m.i_sq[l, i, j])**2
    rhs = (m.v_sq[i] + m.i_sq[l, i, j])**2
    return lhs <= rhs

# (8) Flow Bounds for candidate (l,i,j):
def current_limit_rule(m, l, i, j):
    return pyo.inequality(-m.i_max[i]**2, m.i_sq[l, i, j], m.i_max[i]**2)
    
# (2) Voltage Limits: enforce v_sq[i] in [vmin^2, vmax^2].
def voltage_limits_rule(m, i):
        return pyo.inequality(m.vmin[i]**2, m.v_sq[i], m.vmax[i]**2)

penalty_coef = 1e6
def objective_rule(m):
    loss_term = sum(m.r[l] * m.i_sq[l, i, j] for (l, i, j) in m.LC)
    slack_penalty = penalty_coef * sum(m.s[l, i, j] for (l, i, j) in m.LC)
    return loss_term + slack_penalty