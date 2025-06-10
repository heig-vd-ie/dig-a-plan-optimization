import pyomo.environ as pyo

def slave_model_variables(model: pyo.AbstractModel) -> pyo.AbstractModel:
    
    # Candidate-indexed branch variables.
    model.p_flow = pyo.Var(model.LC, domain=pyo.Reals)
    model.q_flow = pyo.Var(model.LC, domain=pyo.Reals)
    model.i_sq = pyo.Var(model.LC, domain=pyo.NonNegativeReals)
    model.v_sq = pyo.Var(model.N, domain=pyo.NonNegativeReals) # type: ignore
    # model.d = pyo.Var(model.LC, domain=pyo.Binary)
    model.d = pyo.Var(model.LC, domain=pyo.NonNegativeReals, bounds=(0,1))
    # Slack variable for voltage drop constraints.
    # model.slack_v_sq = pyo.Var(model.N, domain=pyo.NonNegativeReals)
    model.slack_v_pos = pyo.Var(model.N, domain=pyo.NonNegativeReals)
    model.slack_v_neg = pyo.Var(model.N, domain=pyo.NonNegativeReals)
    model.slack_i_sq = pyo.Var(model.LC, domain=pyo.NonNegativeReals)
    
    return model