import pyomo.environ as pyo


def master_model_parameters(model: pyo.AbstractModel) -> pyo.AbstractModel:
    model.r = pyo.Param(model.L)  # Resistance for branch l.
    model.x = pyo.Param(model.L)  # Reactance for branch l.
    model.b = pyo.Param(model.L)  # Susceptance for branch l.
    model.p_node = pyo.Param(model.N)  # Real node at bus i.
    model.q_node = pyo.Param(model.N)  # Reactive load at bus i.

    model.slack_node = pyo.Param()  # Slack bus index.
    model.slack_node_v_sq = pyo.Param()  # e.g. 1.0
    model.v_min = pyo.Param(model.N)  # per-unit
    model.v_max = pyo.Param(model.N)
    model.n_transfo = pyo.Param(model.C)  # Transformer turn ration in pu for branch l.
    model.big_m = pyo.Param()  # Big M value for constraints.
    model.small_m = pyo.Param()  # Small M value for constraints.

    return model
