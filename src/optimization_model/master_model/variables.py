# variables.py
import pyomo.environ as pyo

# from shapely import bounds


def master_model_variables(
    model: pyo.AbstractModel, relaxed: bool = False
) -> pyo.AbstractModel:

    if relaxed:
        model.δ = pyo.Var(
            model.S, domain=pyo.Reals, bounds=(0, 1)
        )  # Default value for δ, mutable for testing.
    else:
        model.δ = pyo.Var(
            model.S, domain=pyo.Binary
        )  # Default value for δ, mutable for testing.
    model.flow = pyo.Var(model.C, domain=pyo.Reals)

    model.p_flow = pyo.Var(model.C, domain=pyo.Reals)
    model.q_flow = pyo.Var(model.C, domain=pyo.Reals)
    model.v_sq = pyo.Var(model.N, domain=pyo.NonNegativeReals)
    model.p_slack_node = pyo.Var(domain=pyo.Reals)
    model.q_slack_node = pyo.Var(domain=pyo.Reals)

    model.theta = pyo.Var(domain=pyo.Reals, bounds=(-1, None))  # Bender cuts.

    return model
