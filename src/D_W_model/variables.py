import pyomo.environ as pyo
from shapely import bounds

def D_W_model_variables(model: pyo.AbstractModel) -> pyo.AbstractModel:

    model.delta = pyo.Var(model.S, domain=pyo.Binary)
    model.p_flow= pyo.Var(model.LC, domain=pyo.Reals)
    model.q_flow = pyo.Var(model.LC, domain=pyo.Reals)
    # --- Dantzig–Wolfe columns -------
    model.K          = pyo.Set(initialize=[])                # column indices
    model.lambda_k   = pyo.Var(model.K, domain=pyo.NonNegativeReals)
    # column_d[k,l,i,j] will hold the 0/1 pattern of each column k
    model.column_d   = pyo.Param(model.K, model.LC, default=0, mutable=True)
    model.column_cost= pyo.Param(model.K, initialize=lambda m,k: m._col_cost[k])
    
    return model