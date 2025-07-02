from attr import mutable
import pyomo.environ as pyo
from traitlets import default

def master_model_parameters(model: pyo.AbstractModel) -> pyo.AbstractModel:
    model.r = pyo.Param(model.L)         # Resistance for branch l.
    model.x = pyo.Param(model.L)         # Reactance for branch l.
    model.p_node = pyo.Param(model.N)         # Real node at bus i.
    model.q_node = pyo.Param(model.N)         # Reactive load at bus i.
    model.big_m = pyo.Param()    # Big-M constant.
    model.slack_node = pyo.Param()     # Slack bus index.
    model.slack_node_v_sq = pyo.Param()       # e.g. 1.0
    model.v_min        = pyo.Param(model.N)   # per-unit
    model.v_max        = pyo.Param(model.N)
    model.n_transfo = pyo.Param(model.LC, default=1)  # Transformer turn ration in pu for branch l. 
    model.penalty_cost = pyo.Param()

    return model

def test_master_model_parameters(model: pyo.AbstractModel) -> pyo.AbstractModel:
    model.big_m = pyo.Param()    # Big-M constant.
    model.slack_node = pyo.Param()     # Slack bus index.
    
    model.delta = pyo.Param(model.S, default=1, mutable=True)  # Default value for delta, mutable for testing.

    return model