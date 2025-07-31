import pyomo.environ as pyo
from traitlets import default


def slave_model_parameters(model: pyo.AbstractModel) -> pyo.AbstractModel:
    model.number_of_lines = pyo.Param()  # Total number of lines
    model.r = pyo.Param(model.L)  # Resistance in pu for branch l.
    model.x = pyo.Param(model.L)  # Reactance in pu for branch l.
    model.b = pyo.Param(model.L)  # Shunt susceptance in pu for branch l.
    model.n_transfo = pyo.Param(model.C)  # Transformer turn ration in pu for branch l.
    model.p_node = pyo.Param(model.N)  # Real node at bus i.
    model.q_node = pyo.Param(model.N)  # Reactive load at bus i.
    model.i_max = pyo.Param(model.L)  # Minimum current (p.u.)
    # Soft defaults for voltage limits if none are passed in:
    model.v_min = pyo.Param(model.N, default=0.95)  # Minimum voltage (p.u.)
    model.v_max = pyo.Param(model.N, default=1.05)  # Maximum voltage (p.u.)
    # master_d is defined over LF: 1 if candidate is active, else 0.
    model.master_δ = pyo.Param(model.S, default=0, mutable=True)
    model.master_d = pyo.Param(model.C, default=0, mutable=True)

    model.slack_node_v_sq = pyo.Param(default=1.0)  # Slack bus voltage squared (p.u.)

    # Big-M and penalty costs: give safe defaults and allow mutation
    model.big_m = pyo.Param()
    # model.penalty_cost = pyo.Param()

    return model
