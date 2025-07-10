import pyomo.environ as pyo

def master_model_sets(model: pyo.AbstractModel) -> pyo.AbstractModel:
    model.N = pyo.Set()               # Nodes indices.
    model.L = pyo.Set()               # Edges indices.
    model.S = pyo.Set(within=model.L) # Switch indices
    
    model.NG = pyo.Set(model.N, within=model.N)  
    model.C = pyo.Set(dimen=3,  within=model.L*model.N*model.N) # type: ignore
    # model.S_C = pyo.Set(dimen=2, within=model.N*model.N) # type: ignore
    model.nS = pyo.Set(initialize=lambda m: [l for l in m.L if l not in m.S])
    
    
    return model