import pyomo.environ as pyo
from traitlets import default


def master_model_parameters(model: pyo.AbstractModel) -> pyo.AbstractModel:
    model.number_of_lines = pyo.Param()  # Total number of lines
    model.r = pyo.Param(model.L)  # Resistance for branch l.
    model.x = pyo.Param(model.L)  # Reactance for branch l.
    model.b = pyo.Param(model.L)  # Susceptance for branch l.
    model.p_node = pyo.Param(model.N)  # Real node at bus i.
    model.q_node = pyo.Param(model.N)  # Reactive load at bus i.

    model.slack_node_v_sq = pyo.Param(default=1.0)  # Slack bus voltage squared (p.u.)
    model.v_min = pyo.Param(model.N, default=0.95)  # Minimum voltage (p.u.)
    model.v_max = pyo.Param(model.N, default=1.05)  # Maximum voltage (p.u.)
    model.n_transfo = pyo.Param(
        model.C, default=1.0
    )  # Transformer turn ratio in pu for branch l.
    model.big_m = pyo.Param(default=1e6)  # Big M value for constraints.
    model.small_m = pyo.Param(default=1e-6)  # Small M value for constraints.

    return model
