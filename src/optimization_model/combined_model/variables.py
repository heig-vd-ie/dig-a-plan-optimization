import pyomo.environ as pyo


def model_variables(model: pyo.AbstractModel) -> pyo.AbstractModel:
    # Flow orientation for radiality
    model.flow = pyo.Var(model.C, domain=pyo.Reals)
    # Switch binary variables for topology
    model.δ = pyo.Var(model.S, domain=pyo.Binary)
    model.ζ = pyo.Var(model.TrTaps, domain=pyo.Binary)
    model.i_sq = pyo.Var(model.CΩ, domain=pyo.NonNegativeReals)

    return model
