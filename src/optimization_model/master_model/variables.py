# variables.py
import pyomo.environ as pyo
from shapely import bounds


def master_model_variables(model: pyo.AbstractModel) -> pyo.AbstractModel:


    model.delta = pyo.Var(model.S, domain=pyo.Binary)
    model.d = pyo.Var(model.C, domain=pyo.Reals)

    model.theta = pyo.Var(domain=pyo.Reals, bounds=(-1e8, None)) # Bender cuts.

    return model

# def test_master_model_variables(model: pyo.AbstractModel) -> pyo.AbstractModel:

#     model.d = pyo.Var(model.LC, domain=pyo.Binary)

#     model.p_flow= pyo.Var(model.LC, domain=pyo.Reals)
    
#     model.theta = pyo.Var(domain=pyo.Reals, bounds=(-1e8, None)) # Bender cuts.
#     return model
