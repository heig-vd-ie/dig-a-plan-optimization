import pyomo.environ as pyo
from model_reconfiguration.constraints import *


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
    model.installed_cons = pyo.Constraint(model.NΩ, rule=installed_cons_rule)
    model.installed_prod = pyo.Constraint(model.NΩ, rule=installed_prod_rule)
    model.current_balance = pyo.Constraint(model.CΩ, rule=current_balance_rule)
    model.p_curt_cons_rule = pyo.Constraint(model.NΩ, rule=node_active_power_rule)
    model.q_curt_cons_rule = pyo.Constraint(model.NΩ, rule=node_reactive_power_rule)
    model.p_curt_prod_rule = pyo.Constraint(model.NΩ, rule=node_active_power_prod_rule)
    model.q_curt_prod_rule = pyo.Constraint(
        model.NΩ, rule=node_reactive_power_prod_rule
    )

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
