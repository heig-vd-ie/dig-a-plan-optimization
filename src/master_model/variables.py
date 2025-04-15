# variables.py
import pyomo.environ as pyo

def master_model_variables(model: pyo.AbstractModel) -> pyo.AbstractModel:

    model.d = pyo.Var(model.LC, domain=pyo.Binary)
    model.Delta = pyo.Var(model.S, domain=pyo.Binary)
    model.P = pyo.Var(model.LC, domain=pyo.Reals)
    model.Q = pyo.Var(model.LC, domain=pyo.Reals)

    return model
