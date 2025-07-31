import pyomo.environ as pyo


def master_model_sets(model: pyo.AbstractModel) -> pyo.AbstractModel:
    model.slack_node = pyo.Set()
    model.N = pyo.Set()  # Nodes indices.
    model.L = pyo.Set()  # Edges indices.
    model.S = pyo.Set(within=model.L)  # Switch indices

    model.C = pyo.Set(dimen=3, within=model.L * model.N * model.N)  # type: ignore

    model.Cs = pyo.Set(initialize=lambda m: [(l, i, j) for l, i, j in m.C if l in m.S])
    model.Cl = pyo.Set(
        initialize=lambda m: [(l, i, j) for l, i, j in m.C if l not in m.S]
    )
    return model
