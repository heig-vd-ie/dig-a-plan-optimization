from model_reconfiguration.constraints import *
import pyomo.environ as pyo


def combined_model_common_constraints(model: pyo.AbstractModel) -> pyo.AbstractModel:
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
    model.voltage_limit_lower = pyo.Constraint(model.CsΩ, rule=voltage_limit_lower_rule)
    model.voltage_limit_upper = pyo.Constraint(model.CsΩ, rule=voltage_limit_upper_rule)

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
    model.current_balance = pyo.Constraint(model.CΩ, rule=current_balance_rule)
    model.node_active_power = pyo.Constraint(model.NΩ, rule=node_active_power_rule)
    model.node_active_power_prod = pyo.Constraint(
        model.NΩ, rule=node_active_power_prod_rule
    )
    model.current_limit = pyo.Constraint(model.ClΩ, rule=optimal_current_limit_rule)
    model.current_limit_tr = pyo.Constraint(model.CtΩ, rule=optimal_current_limit_rule)
    model.objective = pyo.Objective(rule=objective_rule_combined, sense=pyo.minimize)
    model.voltage_upper_limits = pyo.Constraint(
        model.NΩ, rule=optimal_voltage_upper_limits_rule
    )
    model.voltage_lower_limits = pyo.Constraint(
        model.NΩ, rule=optimal_voltage_lower_limits_rule
    )
    return model


def combined_model_constraints(model: pyo.AbstractModel) -> pyo.AbstractModel:
    """Adds the constraints to the combined model."""
    model.voltage_drop_line = pyo.Constraint(model.ClΩ, rule=voltage_drop_line_rule)
    model.voltage_drop_transfo = pyo.Constraint(
        model.CtΩ, rule=voltage_drop_transfo_rule
    )
    model.voltage_tap_upper_limit = pyo.Constraint(
        model.CttapΩ, rule=voltage_tap_upper_limit_rule
    )
    model.voltage_tap_lower_limit = pyo.Constraint(
        model.CttapΩ, rule=voltage_tap_lower_limit_rule
    )
    model.tap_limit = pyo.Constraint(model.Tr, rule=tap_limit_rule)
    model.current_rotated_cone = pyo.Constraint(
        model.ClΩ, rule=current_rotated_cone_rule
    )
    model.current_rotated_cone_transformer = pyo.Constraint(
        model.CtΩ, rule=current_rotated_cone_transformer_rule
    )
    return model


def combined_model_lin_constraints(model: pyo.AbstractModel) -> pyo.AbstractModel:
    model.voltage_drop_line = pyo.Constraint(
        model.ClΩ, rule=voltage_drop_line_lindistflow_rule
    )
    model.voltage_drop_transfo = pyo.Constraint(
        model.CtΩ, rule=voltage_drop_transfo_lindistflow_rule
    )
    return model
