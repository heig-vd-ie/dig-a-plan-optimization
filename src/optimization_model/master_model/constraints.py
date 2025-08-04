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

- If the branch :math:`l` is switchable (i.e., :math:`l \in S`), then its activation depends on the switch status :math:`\δ_l`.
- If the branch is not switchable, then exactly one candidate connection must be selected.

.. math::
    :label: master-orientation
    :nowrap:

    \begin{align}
        d_{l~i~j} + d_{l~j~i} =
        \begin{cases}
            \δ_l & \text{if } l \in S \\
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
from optimization_model.constraints import *


def master_model_constraints(model: pyo.AbstractModel) -> pyo.AbstractModel:

    model.objective = pyo.Objective(rule=master_obj, sense=pyo.minimize)

    model.flow_balance = pyo.Constraint(model.Nes, rule=imaginary_flow_balance_rule)
    model.flow_balance_slack = pyo.Constraint(
        model.slack_node, rule=imaginary_flow_balance_slack_rule
    )
    model.edge_propagation = pyo.Constraint(
        model.L, rule=imaginary_flow_edge_propagation_rule
    )

    model.upper_switch_propagation = pyo.Constraint(
        model.Cs, rule=imaginary_flow_upper_switch_propagation_rule
    )
    model.lower_switch_propagation = pyo.Constraint(
        model.Cs, rule=imaginary_flow_lower_switch_propagation_rule
    )
    model.nb_closed_switches = pyo.Constraint(
        rule=imaginary_flow_nb_closed_switches_rule
    )

    model.edge_active_power_balance = pyo.Constraint(
        model.LΩ, rule=edge_active_power_balance_lindistflow_rule
    )
    model.edge_reactive_power_balance = pyo.Constraint(
        model.LΩ, rule=edge_reactive_power_balance_lindistflow_rule
    )
    model.node_active_power_balance = pyo.Constraint(
        model.NesΩ, rule=node_active_power_balance_rule
    )
    model.node_active_power_balance_slack = pyo.Constraint(
        model.slack_nodeΩ, rule=node_active_power_balance_slack_rule
    )
    model.node_reactive_power_balance = pyo.Constraint(
        model.NesΩ, rule=node_reactive_power_balance_rule
    )
    model.node_reactive_power_balance_slack = pyo.Constraint(
        model.slack_nodeΩ, rule=node_reactive_power_balance_slack_rule
    )
    ##
    model.slack_voltage = pyo.Constraint(model.slack_nodeΩ, rule=slack_voltage_rule)
    model.voltage_drop_lower = pyo.Constraint(model.CsΩ, rule=voltage_drop_lower_rule)
    model.voltage_drop_upper = pyo.Constraint(model.CsΩ, rule=voltage_drop_upper_rule)
    model.voltage_drop_line = pyo.Constraint(
        model.ClΩ, rule=voltage_drop_line_lindistflow_rule
    )
    model.voltage_upper_limits = pyo.Constraint(
        model.NΩ, rule=optimal_voltage_upper_limits_distflow_rule
    )
    model.voltage_lower_limits = pyo.Constraint(
        model.NΩ, rule=optimal_voltage_lower_limits_distflow_rule
    )
    model.switch_active_power_lower_bound = pyo.Constraint(
        model.CsΩ, rule=switch_active_power_lower_bound_rule
    )
    model.switch_active_power_upper_bound = pyo.Constraint(
        model.CsΩ, rule=switch_active_power_upper_bound_rule
    )
    #
    model.switch_reactive_power_lower_bound = pyo.Constraint(
        model.CsΩ, rule=switch_reactive_power_lower_bound_rule
    )
    model.switch_reactive_power_upper_bound = pyo.Constraint(
        model.CsΩ, rule=switch_reactive_power_upper_bound_rule
    )
    # cuts are generated on-the-fly, so no rules are necessary.
    model.infeasibility_cut = ConstraintList()
    model.optimality_cut = ConstraintList()

    return model
