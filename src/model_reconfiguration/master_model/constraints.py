# constraints.py
import pyomo.environ as pyo
import pyomo.gdp as pyg
from pyomo.environ import ConstraintList
from model_reconfiguration.constraints import *


def master_model_constraints(model: pyo.AbstractModel) -> pyo.AbstractModel:

    model.objective = pyo.Objective(rule=master_obj, sense=pyo.minimize)

    model.flow_balance = pyo.Constraint(model.Nes, rule=imaginary_flow_balance_rule)
    model.flow_balance_slack = pyo.Constraint(
        model.slack_node, rule=imaginary_flow_balance_slack_rule
    )
    model.edge_propagation = pyo.Constraint(
        model.E, rule=imaginary_flow_edge_propagation_rule
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
        model.EΩ, rule=edge_active_power_balance_lindistflow_rule
    )
    model.edge_reactive_power_balance = pyo.Constraint(
        model.EΩ, rule=edge_reactive_power_balance_lindistflow_rule
    )
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
    ##
    model.slack_voltage = pyo.Constraint(model.snΩ, rule=slack_voltage_rule)
    model.voltage_limit_lower = pyo.Constraint(model.CsΩ, rule=voltage_limit_lower_rule)
    model.voltage_limit_upper = pyo.Constraint(model.CsΩ, rule=voltage_limit_upper_rule)
    model.voltage_drop_line = pyo.Constraint(
        model.ClΩ, rule=voltage_drop_line_lindistflow_rule
    )
    model.voltage_drop_transfo = pyo.Constraint(
        model.CtΩ, rule=voltage_drop_transfo_lindistflow_rule
    )
    model.voltage_tap_lower_limit = pyo.Constraint(
        model.CttapΩ, rule=voltage_tap_lower_limit_rule
    )
    model.voltage_tap_upper_limit = pyo.Constraint(
        model.CttapΩ, rule=voltage_tap_upper_limit_rule
    )
    model.tap_limit_rule = pyo.Constraint(model.Tr, rule=tap_limit_rule)
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
    model.installed_cons = pyo.Constraint(model.NΩ, rule=installed_cons_rule)
    model.installed_prod = pyo.Constraint(model.NΩ, rule=installed_prod_rule)
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
    # cuts are generated on-the-fly, so no rules are necessary.
    model.infeasibility_cut = ConstraintList()
    model.optimality_cut = ConstraintList()

    return model
