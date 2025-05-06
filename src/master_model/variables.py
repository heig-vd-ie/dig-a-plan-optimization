# variables.py
import pyomo.environ as pyo

def master_model_variables(model: pyo.AbstractModel) -> pyo.AbstractModel:

    model.d = pyo.Var(model.LC, domain=pyo.Binary)
    model.delta = pyo.Var(model.S, domain=pyo.Binary)
    model.p_flow= pyo.Var(model.LC, domain=pyo.Reals)
    model.q_flow = pyo.Var(model.LC, domain=pyo.Reals)
    
    # Benders auxiliary variable
    model.Theta = pyo.Var(domain=pyo.NonNegativeReals)


    return model
