# variables.py
import pyomo.environ as pyo
from shapely import bounds


def master_model_variables(model: pyo.AbstractModel) -> pyo.AbstractModel:

    model.d = pyo.Var(model.LC, domain=pyo.Binary)
    model.delta = pyo.Var(model.S, domain=pyo.Binary)
    model.p_flow= pyo.Var(model.LC, domain=pyo.Reals)
    model.q_flow = pyo.Var(model.LC, domain=pyo.Reals)
    
    model.losses = pyo.Var(domain=pyo.Reals) # Bender cuts.
    model.theta = pyo.Var(domain=pyo.Reals, bounds=(-1e-8, None)) # Bender cuts.
    model.v_sq = pyo.Var(model.N, domain=pyo.NonNegativeReals)
    
    model.slack_v_pos = pyo.Var(model.N, domain=pyo.NonNegativeReals)
    model.slack_v_neg = pyo.Var(model.N, domain=pyo.NonNegativeReals)


    return model
