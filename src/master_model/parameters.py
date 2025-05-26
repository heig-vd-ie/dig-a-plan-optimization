import pyomo.environ as pyo

def master_model_parameters(model: pyo.AbstractModel) -> pyo.AbstractModel:
    model.r = pyo.Param(model.L)         # Resistance for branch l.
    model.x = pyo.Param(model.L)         # Reactance for branch l.
    model.p_node = pyo.Param(model.N)         # Real node at bus i.
    model.q_node = pyo.Param(model.N)         # Reactive load at bus i.
    model.big_m = pyo.Param()    # Big-M constant.
    model.slack_node = pyo.Param()     # Slack bus index.
    model.slave_objective = pyo.Param(default= 0, mutable=True)  # Slave model objective value.
    model.marginal_cost = pyo.Param(model.LC, default= 0, mutable=True)  # Marginal cost of the system.
    model.previous_d = pyo.Param(model.LC, default= 0, mutable=True) # Previous d candidate status.
    return model