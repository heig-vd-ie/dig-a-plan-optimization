import pyomo.environ as pyo


def model_sets(model: pyo.AbstractModel) -> pyo.AbstractModel:
    model.SCEN = pyo.Set() # Scenario index
    # Node, line, switch and candidate arc definitions
    model.N = pyo.Set()  # Nodes indices
    model.L = pyo.Set()  # Edges indices
    model.S = pyo.Set(within=model.L)  # Switch indices
    model.C = pyo.Set(dimen=3, within=model.L * model.N * model.N)  # type: ignore
    # Non-switch lines
    model.nS = pyo.Set(initialize=lambda m: [l for l in m.L if l not in m.S])

    return model
