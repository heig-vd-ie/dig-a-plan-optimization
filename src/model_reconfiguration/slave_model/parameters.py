import pyomo.environ as pyo
from traitlets import default


def slave_model_parameters(model: pyo.AbstractModel) -> pyo.AbstractModel:
    """Define parameters for the slave model."""
    model.master_δ = pyo.Param(model.S, default=0, mutable=True)
    model.master_ζ = pyo.Param(model.TrTaps, default=0, mutable=True)
    return model
