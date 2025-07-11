# variables.py
import pyomo.environ as pyo

# from shapely import bounds


def master_model_variables(model: pyo.AbstractModel) -> pyo.AbstractModel:

    # model.delta = pyo.Var(model.S, domain=pyo.Reals, bounds=(0, 1))  # Default value for delta, mutable for testing.
    model.delta = pyo.Var(
        model.S, domain=pyo.Binary
    )  # Default value for delta, mutable for testing.

    model.flow = pyo.Var(model.C, domain=pyo.Reals)

    model.theta = pyo.Var(domain=pyo.Reals, bounds=(-1e-5, None))  # Bender cuts.

    return model
