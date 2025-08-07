# variables.py
import pyomo.environ as pyo

# from shapely import bounds


def master_model_variables(
    model: pyo.AbstractModel, relaxed: bool = False
) -> pyo.AbstractModel:
    # Flow orientation for radiality
    model.flow = pyo.Var(model.C, domain=pyo.Reals)
    model.δ = (
        pyo.Var(model.S, domain=pyo.Reals, bounds=(0, 1))
        if relaxed
        else pyo.Var(model.S, domain=pyo.Binary)
    )
    model.ζ = (
        pyo.Var(model.TrTaps, domain=pyo.Reals, bounds=(0, 1))
        if relaxed
        else pyo.Var(model.TrTaps, domain=pyo.Binary)
    )

    model.θ1 = pyo.Var(domain=pyo.Reals, bounds=(-1, None))  # Bender cuts.
    model.θ2 = pyo.Var(domain=pyo.Reals, bounds=(-1, None))  # Bender cuts.

    return model
