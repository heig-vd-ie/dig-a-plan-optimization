import pyomo.environ as pyo
from shapely import bounds

def D_W_model_variables(model: pyo.AbstractModel) -> pyo.AbstractModel:

    model.delta = pyo.Var(model.L, domain=pyo.Binary)
    # model.p_flow= pyo.Var(model.LC, domain=pyo.Reals)
    # model.q_flow = pyo.Var(model.LC, domain=pyo.Reals)
    # --- Dantzig–Wolfe columns -------
    model.lambda_k   = pyo.Var(model.K, domain=pyo.NonNegativeReals)
    
    return model

