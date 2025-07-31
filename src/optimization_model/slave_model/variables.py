import pyomo.environ as pyo


def slave_model_variables(model: pyo.AbstractModel) -> pyo.AbstractModel:

    # Candidate-indexed branch variables.
    model.p_flow = pyo.Var(model.C, domain=pyo.Reals)
    model.q_flow = pyo.Var(model.C, domain=pyo.Reals)
    model.i_sq = pyo.Var(model.C, domain=pyo.NonNegativeReals)
    model.v_sq = pyo.Var(model.N, domain=pyo.NonNegativeReals)

    model.δ = pyo.Var(
        model.S, domain=pyo.Reals, bounds=(0, 1)
    )  # Candidate active (1) or not (0).

    model.p_slack_node = pyo.Var(domain=pyo.Reals)  # Real power at the slack node.
    model.q_slack_node = pyo.Var(domain=pyo.Reals)  # Reactive power at the slack node.

    return model


def infeasible_slave_model_variables(model: pyo.AbstractModel) -> pyo.AbstractModel:
    model = slave_model_variables(model)
    model.slack_v_pos = pyo.Var(model.N, domain=pyo.NonNegativeReals)
    model.slack_v_neg = pyo.Var(model.N, domain=pyo.NonNegativeReals)
    model.slack_i_sq = pyo.Var(model.C, domain=pyo.NonNegativeReals)
    return model
