import pyomo.environ as pyo

def slave_model_sets(model: pyo.AbstractModel) -> pyo.AbstractModel:
    model.N = pyo.Set()               # Bus indices.
    model.L = pyo.Set()               # Branch indices.
    # C: for each branch l, candidate bus pairs.
    model.C = pyo.Set(model.L, within=model.N * model.N) # type: ignore
    # Candidate connectivity set LC: all tuples (l, i, j) for each (i, j) in C[l].
    model.LC = pyo.Set(
        dimen=3,
        initialize=lambda m: [(l, i, j) for l in m.L for (i, j) in m.C[l]] + [(l, j, i) for l in m.L for (i, j) in m.C[l]]
    )
    
    model.S = pyo.Set(within=model.L)
    model.nS = pyo.Set(initialize=lambda m: [l for l in m.L if l not in m.S])
    
    return model