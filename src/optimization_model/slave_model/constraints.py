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
from optimization_model.constraints import *


def slave_model_constraints(model: pyo.AbstractModel) -> pyo.AbstractModel:
    model.master_switch_status_propagation = pyo.Constraint(
        model.S, rule=master_switch_status_propagation_rule
    )
    model.master_transformer_status_propagation = pyo.Constraint(
        model.TrTaps, rule=master_transformer_status_propagation_rule
    )
    # Distflow equations
    model.slack_voltage = pyo.Constraint(model.snΩ, rule=slack_voltage_rule)
    model.node_active_power_balance = pyo.Constraint(
        model.NesΩ, rule=node_active_power_balance_rule
    )
    model.node_active_power_balance_slack = pyo.Constraint(
        model.snΩ, rule=node_active_power_balance_slack_rule
    )
    model.node_reactive_power_balance = pyo.Constraint(
        model.NesΩ, rule=node_reactive_power_balance_rule
    )
    model.node_reactive_power_balance_slack = pyo.Constraint(
        model.snΩ, rule=node_reactive_power_balance_slack_rule
    )
    model.voltage_limit_lower = pyo.Constraint(model.CsΩ, rule=voltage_limit_lower_rule)
    model.voltage_limit_upper = pyo.Constraint(model.CsΩ, rule=voltage_limit_upper_rule)
    model.voltage_drop_line = pyo.Constraint(model.ClΩ, rule=voltage_drop_line_rule)
    model.voltage_drop_transfo = pyo.Constraint(
        model.CtΩ, rule=voltage_drop_transfo_rule
    )

    model.current_rotated_cone = pyo.Constraint(
        model.ClΩ, rule=current_rotated_cone_rule
    )
    model.current_rotated_cone_transformer = pyo.Constraint(
        model.CtΩ, rule=current_rotated_cone_transformer_rule
    )
    model.edge_active_power_balance = pyo.Constraint(
        model.SΩ, rule=edge_active_power_balance_switch_rule
    )
    model.edge_reactive_power_balance = pyo.Constraint(
        model.SΩ, rule=edge_reactive_power_balance_switch_rule
    )
    model.edge_active_power_balance_line = pyo.Constraint(
        model.EΩ, rule=edge_active_power_balance_line_rule
    )
    model.edge_reactive_power_balance_line = pyo.Constraint(
        model.EΩ, rule=edge_reactive_power_balance_line_rule
    )
    # Switch status constraints
    model.switch_active_power_lower_bound = pyo.Constraint(
        model.CsΩ, rule=switch_active_power_lower_bound_rule
    )
    model.switch_active_power_upper_bound = pyo.Constraint(
        model.CsΩ, rule=switch_active_power_upper_bound_rule
    )
    model.switch_reactive_power_lower_bound = pyo.Constraint(
        model.CsΩ, rule=switch_reactive_power_lower_bound_rule
    )
    model.switch_reactive_power_upper_bound = pyo.Constraint(
        model.CsΩ, rule=switch_reactive_power_upper_bound_rule
    )
    model.tap_upper_limit = pyo.Constraint(
        model.CttapΩ, rule=voltage_tap_upper_limit_rule
    )
    model.tap_lower_limit = pyo.Constraint(
        model.CttapΩ, rule=voltage_tap_lower_limit_rule
    )
    model.tap_limit = pyo.Constraint(model.Tr, rule=tap_limit_rule)
    model.current_balance = pyo.Constraint(model.CΩ, rule=current_balance_rule)

    return model


def optimal_slave_model_constraints(model: pyo.AbstractModel) -> pyo.AbstractModel:
    model = slave_model_constraints(model)
    model.objective = pyo.Objective(rule=objective_rule_loss, sense=pyo.minimize)
    # Physical limits
    model.current_limit = pyo.Constraint(model.ClΩ, rule=optimal_current_limit_rule)
    model.current_limit_transformer = pyo.Constraint(
        model.CtΩ, rule=optimal_current_limit_rule
    )
    model.voltage_upper_limits = pyo.Constraint(
        model.NΩ, rule=optimal_voltage_upper_limits_rule
    )
    model.voltage_lower_limits = pyo.Constraint(
        model.NΩ, rule=optimal_voltage_lower_limits_rule
    )
    model.power_curt_cons = pyo.Constraint(
        model.NΩ, rule=(lambda m, n, ω: m.p_curt_cons[n, ω] == 0.0)
    )
    model.power_curt_prod = pyo.Constraint(
        model.NΩ, rule=(lambda m, n, ω: m.p_curt_prod[n, ω] == 0.0)
    )
    model.reactive_power_curt_cons = pyo.Constraint(
        model.NΩ, rule=(lambda m, n, ω: m.q_curt_cons[n, ω] == 0.0)
    )
    model.reactive_power_curt_prod = pyo.Constraint(
        model.NΩ, rule=(lambda m, n, ω: m.q_curt_prod[n, ω] == 0.0)
    )
    return model


def infeasible_slave_model_constraints(model: pyo.AbstractModel) -> pyo.AbstractModel:
    model = slave_model_constraints(model)
    model.objective = pyo.Objective(
        rule=objective_rule_infeasibility, sense=pyo.minimize
    )
    return model
