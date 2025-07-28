import pyomo.environ as pyo


def model_variables(model: pyo.AbstractModel) -> pyo.AbstractModel:
    # Switch binary variables for topology
    model.delta = pyo.Var(
        model.SCEN, model.S, domain=pyo.NonNegativeReals, bounds=(0, 1)
    )  # switch status per scenario (continuous relaxation)
    model.delta_penalty = pyo.Var(model.SCEN, model.S, domain=pyo.NonNegativeReals)

    # Flow orientation for radiality
    model.flow = pyo.Var(model.SCEN, model.C, domain=pyo.Reals)
    # DistFlow variables
    model.p_flow = pyo.Var(model.SCEN, model.C, domain=pyo.Reals)
    model.q_flow = pyo.Var(model.SCEN, model.C, domain=pyo.Reals)
    model.i_sq = pyo.Var(model.SCEN, model.C, domain=pyo.NonNegativeReals)
    model.v_sq = pyo.Var(model.SCEN, model.N, domain=pyo.NonNegativeReals)
    # Slack injections
    model.p_slack_node = pyo.Var(model.SCEN, domain=pyo.Reals)
    model.q_slack_node = pyo.Var(model.SCEN, domain=pyo.Reals)

    # Relaxation variables
    model.slack_i_sq = pyo.Var(model.SCEN, model.C, domain=pyo.NonNegativeReals)
    model.slack_v_pos = pyo.Var(model.SCEN, model.N, domain=pyo.NonNegativeReals)
    model.slack_v_neg = pyo.Var(model.SCEN, model.N, domain=pyo.NonNegativeReals)

    return model
