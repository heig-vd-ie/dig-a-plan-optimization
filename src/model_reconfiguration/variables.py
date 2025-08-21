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

    model.v_relax_up = pyo.Var(model.NΩ, domain=pyo.NonNegativeReals)
    model.v_relax_down = pyo.Var(model.NΩ, domain=pyo.NonNegativeReals)
    model.node_cons_installed = pyo.Var(model.NΩ, domain=pyo.Reals)
    model.node_prod_installed = pyo.Var(model.NΩ, domain=pyo.Reals)

    return model
