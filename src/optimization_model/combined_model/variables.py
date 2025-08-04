import pyomo.environ as pyo


def model_variables(model: pyo.AbstractModel) -> pyo.AbstractModel:
    # Switch binary variables for topology
    model.δ = pyo.Var(model.S, domain=pyo.Binary)

    # Flow orientation for radiality
    model.flow = pyo.Var(model.C, domain=pyo.Reals)
    # DistFlow variables
    model.p_flow = pyo.Var(model.CΩ, domain=pyo.Reals)
    model.q_flow = pyo.Var(model.CΩ, domain=pyo.Reals)
    model.i_sq = pyo.Var(model.CΩ, domain=pyo.NonNegativeReals)
    model.v_sq = pyo.Var(model.NΩ, domain=pyo.NonNegativeReals)
    # Slack injections
    model.p_slack_node = pyo.Var(model.Ω, domain=pyo.Reals)
    model.q_slack_node = pyo.Var(model.Ω, domain=pyo.Reals)

    # Relaxation variables
    model.slack_i_sq = pyo.Var(model.CΩ, domain=pyo.NonNegativeReals)
    model.slack_v_pos = pyo.Var(model.NΩ, domain=pyo.NonNegativeReals)

    return model
