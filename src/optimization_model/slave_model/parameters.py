import pyomo.environ as pyo
from traitlets import default


def slave_model_parameters(model: pyo.AbstractModel) -> pyo.AbstractModel:
    # master_d is defined over LF: 1 if candidate is active, else 0.
    model.master_δ = pyo.Param(model.S, default=0, mutable=True)
    model.master_d = pyo.Param(model.C, default=0, mutable=True)
    return model
