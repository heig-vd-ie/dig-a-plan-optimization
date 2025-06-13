import pyomo.environ as pyo

def master_model_parameters(model: pyo.AbstractModel) -> pyo.AbstractModel:
    model.r = pyo.Param(model.L)         # Resistance for branch l.
    model.x = pyo.Param(model.L)         # Reactance for branch l.
    model.p_node = pyo.Param(model.N)         # Real node at bus i.
    model.q_node = pyo.Param(model.N)         # Reactive load at bus i.
    model.big_m = pyo.Param()    # Big-M constant.
    model.slack_node = pyo.Param()     # Slack bus index.
    model.V0 = pyo.Param()               # Slack‐bus voltage magnitude (per-unit)
        # Soft defaults for voltage limits if none are passed in:
    model.Vmin = pyo.Param(model.N,   default=0.95)
    model.Vmax = pyo.Param(model.N,   default=1.05)

    return model