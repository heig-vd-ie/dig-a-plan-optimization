import pyomo.environ as pyo

def model_parameters(model: pyo.AbstractModel) -> pyo.AbstractModel:
    model.r = pyo.Param(model.L)         # Resistance in pu for branch l.
    model.x = pyo.Param(model.L)         # Reactance in pu for branch l.
    model.b = pyo.Param(model.L)         # Shunt susceptance in pu for branch l.
    model.n_transfo = pyo.Param(model.LC, default=1)  # Transformer turn ration in pu for branch l. 
    model.p_node = pyo.Param(model.N)         # Real node at bus i.
    model.q_node = pyo.Param(model.N)         # Reactive load at bus i.
    model.i_max = pyo.Param(model.L) # Minimum current (p.u.)    
    # Soft defaults for voltage limits if none are passed in:
    model.v_min = pyo.Param(model.N,   default=0.95)
    model.v_max = pyo.Param(model.N,   default=1.05)  
    model.slack_node_v_sq = pyo.Param()             # Slack bus voltage (p.u.)
    model.slack_node = pyo.Param()                  # Slack bus index.
    
    # Big-M and penalty costs: give safe defaults and allow mutation
    model.big_m = pyo.Param()
    model.penalty_cost = pyo.Param()

    model.current_factor = pyo.Param(default=1)  # Scale factor for the objective function.
    model.voltage_factor = pyo.Param(default=1)  # Scale factor for the objective function.
    model.power_factor = pyo.Param(default=1)    # Scale factor for

    return model