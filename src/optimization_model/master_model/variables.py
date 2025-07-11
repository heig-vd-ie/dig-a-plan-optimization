# variables.py
import pyomo.environ as pyo

# from shapely import bounds


def master_model_variables(
    model: pyo.AbstractModel, relaxed: bool = False
) -> pyo.AbstractModel:

    if relaxed:
        model.delta = pyo.Var(
            model.S, domain=pyo.Reals, bounds=(0, 1)
        )  # Default value for delta, mutable for testing.
    else:
        model.delta = pyo.Var(
            model.S, domain=pyo.Binary
        )  # Default value for delta, mutable for testing.
    model.flow = pyo.Var(model.C, domain=pyo.Reals)

    model.theta = pyo.Var(domain=pyo.Reals, bounds=(-1, None))  # Bender cuts.

    return model
