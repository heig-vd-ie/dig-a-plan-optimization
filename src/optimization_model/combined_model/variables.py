import pyomo.environ as pyo
from shapely import bounds


def model_variables(model: pyo.AbstractModel) -> pyo.AbstractModel:
    # Switch binary variables for topology
    model.delta = pyo.Var(model.S, domain=pyo.Binary)
    model.d = pyo.Var(model.C, domain=pyo.Binary)
    # Flow orientation for radiality
    model.flow = pyo.Var(model.C, domain=pyo.Reals)
    # DistFlow variables
    model.p_flow = pyo.Var(model.C, domain=pyo.Reals)
    model.q_flow = pyo.Var(model.C, domain=pyo.Reals)
    model.i_sq = pyo.Var(model.C, domain=pyo.NonNegativeReals)
    model.v_sq = pyo.Var(model.N, domain=pyo.NonNegativeReals)
    # Slack injections
    model.p_slack_node = pyo.Var(domain=pyo.Reals)
    model.q_slack_node = pyo.Var(domain=pyo.Reals)

    # Relaxation variables
    model.i_sq_relax = pyo.Var(model.C, domain=pyo.NonNegativeReals)
    model.v_sq_relax_pos = pyo.Var(
        model.N, domain=pyo.NonNegativeReals, bounds=(0, None)
    )
    model.v_sq_relax_neg = pyo.Var(
        model.N, domain=pyo.NonNegativeReals, bounds=(0, None)
    )

    return model
