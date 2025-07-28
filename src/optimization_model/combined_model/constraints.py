import numpy as np
from optimization_model.constraints import *
import pyomo.environ as pyo


def combined_model_constraints(model: pyo.AbstractModel) -> pyo.AbstractModel:
    # 1) Radiality: each non‑slack node has one incoming flow (per scenario)
    model.flow_balance = pyo.Constraint(
        model.SCEN, model.N, rule=imaginary_flow_balance_rule
    )
    model.edge_propagation = pyo.Constraint(
        model.SCEN, model.L, rule=imaginary_flow_edge_propagation_rule
    )
    model.upper_switch_propagation = pyo.Constraint(
        model.SCEN, model.C, rule=imaginary_flow_upper_switch_propagation_rule
    )
    model.lower_switch_propagation = pyo.Constraint(
        model.SCEN, model.C, rule=imaginary_flow_lower_switch_propagation_rule
    )
    model.nb_closed_switches = pyo.Constraint(
        model.SCEN, rule=imaginary_flow_nb_closed_switches_rule
    )
    # 2) DistFlow & power balance (per scenario)
    model.slack_voltage = pyo.Constraint(model.SCEN, model.N, rule=slack_voltage_rule)
    model.node_active_power_balance = pyo.Constraint(
        model.SCEN, model.N, rule=node_active_power_balance_rule
    )
    model.node_reactive_power_balance = pyo.Constraint(
        model.SCEN, model.N, rule=node_reactive_power_balance_rule
    )
    model.edge_active_power_balance = pyo.Constraint(
        model.SCEN, model.L, rule=edge_active_power_balance_rule
    )
    model.edge_reactive_power_balance = pyo.Constraint(
        model.SCEN, model.L, rule=edge_reactive_power_balance_rule
    )
    # 3) Voltage‐drop & cone (per scenario)
    model.voltage_drop_lower = pyo.Constraint(
        model.SCEN, model.C, rule=voltage_drop_lower_rule
    )
    model.voltage_drop_upper = pyo.Constraint(
        model.SCEN, model.C, rule=voltage_drop_upper_rule
    )
    model.current_rotated_cone = pyo.Constraint(
        model.SCEN, model.C, rule=current_rotated_cone_rule
    )
    # 4) Switch power bounds (per scenario)
    model.switch_active_power_lower_bound = pyo.Constraint(
        model.SCEN, model.C, rule=switch_active_power_lower_bound_rule
    )
    model.switch_active_power_upper_bound = pyo.Constraint(
        model.SCEN, model.C, rule=switch_active_power_upper_bound_rule
    )
    model.switch_reactive_power_lower_bound = pyo.Constraint(
        model.SCEN, model.C, rule=switch_reactive_power_lower_bound_rule
    )
    model.switch_reactive_power_upper_bound = pyo.Constraint(
        model.SCEN, model.C, rule=switch_reactive_power_upper_bound_rule
    )
    # 5) Current symmetry (per scenario)
    model.current_balance = pyo.Constraint(
        model.SCEN, model.C, rule=current_balance_rule
    )
    # 6) Infeasible relaxations (per scenario)
    model.current_limit = pyo.Constraint(
        model.SCEN, model.C, rule=infeasible_current_limit_rule
    )
    model.voltage_upper_limits = pyo.Constraint(
        model.SCEN, model.N, rule=infeasible_voltage_upper_limits_rule
    )
    model.voltage_lower_limits = pyo.Constraint(
        model.SCEN, model.N, rule=infeasible_voltage_lower_limits_rule
    )

    breakpoints = np.linspace(0, 1, 20).tolist()
    values = [x * (1 - x) for x in breakpoints]

    model.piecewise_penalty = pyo.Piecewise(
        model.SCEN,
        model.S,
        model.delta_penalty,
        model.delta,
        pw_pts=breakpoints,
        f_rule=values,
        pw_constr_type="EQ",
        pw_repn="SOS2",
    )

    # model.objective = pyo.Objective(rule=objective_rule_combined, sense=pyo.minimize)
    model.objective = pyo.Objective(rule=ADMM_objective_rule, sense=pyo.minimize)
    return model
