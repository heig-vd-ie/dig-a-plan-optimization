from optimization_model.constraints import *
import pyomo.environ as pyo


def combined_model_constraints(model: pyo.AbstractModel) -> pyo.AbstractModel:
    # Radiality: each non-slack node has one incoming flow
    model.flow_balance = pyo.Constraint(model.N, rule=imaginary_flow_balance_rule)
    model.edge_propagation = pyo.Constraint(
        model.L, rule=imaginary_flow_edge_propagation_rule
    )
    model.upper_switch_propagation = pyo.Constraint(
        model.C, rule=imaginary_flow_upper_switch_propagation_rule
    )
    model.lower_switch_propagation = pyo.Constraint(
        model.C, rule=imaginary_flow_lower_switch_propagation_rule
    )
    model.nb_closed_switches = pyo.Constraint(
        rule=imaginary_flow_nb_closed_switches_rule
    )
    # DistFlow and power balance
    model.slack_voltage = pyo.Constraint(model.N, rule=slack_voltage_rule)
    model.node_active_power_balance = pyo.Constraint(
        model.N, rule=node_active_power_balance_rule
    )
    model.node_reactive_power_balance = pyo.Constraint(
        model.N, rule=node_reactive_power_balance_rule
    )
    model.edge_active_power_balance = pyo.Constraint(
        model.L, rule=edge_active_power_balance_rule
    )
    model.edge_reactive_power_balance = pyo.Constraint(
        model.L, rule=edge_reactive_power_balance_rule
    )
    model.voltage_drop_lower = pyo.Constraint(model.C, rule=voltage_drop_lower_rule)
    model.voltage_drop_upper = pyo.Constraint(model.C, rule=voltage_drop_upper_rule)
    model.current_rotated_cone = pyo.Constraint(model.C, rule=current_rotated_cone_rule)
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
    # Physical limits and objective
    model.current_limit = pyo.Constraint(model.C, rule=infeasible_current_limit_rule)
    model.voltage_upper_limits = pyo.Constraint(
        model.N, rule=infeasible_voltage_upper_limits_rule
    )
    model.voltage_lower_limits = pyo.Constraint(
        model.N, rule=infeasible_voltage_lower_limits_rule
    )
    model.objective = pyo.Objective(
        rule=objective_rule_infeasibility, sense=pyo.minimize
    )
    return model
