import pyomo.environ as pyo


def model_variables(model: pyo.AbstractModel) -> pyo.AbstractModel:
    # DistFlow variables
    model.p_flow = pyo.Var(model.CΩ, domain=pyo.Reals)
    model.q_flow = pyo.Var(model.CΩ, domain=pyo.Reals)
    model.v_sq = pyo.Var(model.NΩ, domain=pyo.NonNegativeReals)
    model.vt_sq = pyo.Var(model.NtapΩ, domain=pyo.NonNegativeReals)
    # Slack injections
    model.p_slack_node = pyo.Var(model.Ω, domain=pyo.Reals)
    model.q_slack_node = pyo.Var(model.Ω, domain=pyo.Reals)

    model.p_curt_cons = pyo.Var(model.NΩ, domain=pyo.NonNegativeReals)
    model.q_curt_cons = pyo.Var(model.NΩ, domain=pyo.NonNegativeReals)
    model.p_curt_prod = pyo.Var(model.NΩ, domain=pyo.NonNegativeReals)
    model.q_curt_prod = pyo.Var(model.NΩ, domain=pyo.NonNegativeReals)

    return model
