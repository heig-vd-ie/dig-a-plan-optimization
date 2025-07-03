

import pyomo.environ as pyo

def model_variables(model: pyo.AbstractModel) -> pyo.AbstractModel:
    
    # Candidate-indexed branch variables.
    # model.d = pyo.Var(model.LC, domain=pyo.Binary)
    model.d = pyo.Var(model.LC, domain=pyo.Binary)  # Allow fractional candidates for linear model.
    model.delta = pyo.Var(model.S, domain=pyo.Binary)
    
    model.p_flow = pyo.Var(model.LC, domain=pyo.Reals)
    model.q_flow = pyo.Var(model.LC, domain=pyo.Reals)
    model.i_sq = pyo.Var(model.LC, domain=pyo.NonNegativeReals)
    model.v_sq = pyo.Var(model.N, domain=pyo.NonNegativeReals) 
    model.slack_v_pos = pyo.Var(model.N, domain=pyo.NonNegativeReals)
    model.slack_v_neg = pyo.Var(model.N, domain=pyo.NonNegativeReals)
    model.slack_i_sq = pyo.Var(model.LC, domain=pyo.NonNegativeReals)
    return model

