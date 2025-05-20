import pyomo.environ as pyo

def D_W_model_parameters(model: pyo.AbstractModel) -> pyo.AbstractModel:
    model.r = pyo.Param(model.L)         # Resistance for branch l.
    model.x = pyo.Param(model.L)         # Reactance for branch l.
    model.p_node = pyo.Param(model.N)         # Real node at bus i.
    model.q_node = pyo.Param(model.N)         # Reactive load at bus i.
    model.big_m = pyo.Param()    # Big-M constant.
    model.slack_node = pyo.Param()     # Slack bus index.
    # Data for each column k
    model.column_d    = pyo.Param(model.K, model.LC,
                                  within=pyo.Binary,
                                  default=0,
                                  mutable=True)  # d^k[l,i,j]
    model.column_cost = pyo.Param(model.K,
                                 default=0,
                                 mutable=True)
    
    return model

