import pyomo.environ as pyo

def slave_model_parameters(model: pyo.AbstractModel) -> pyo.AbstractModel:
    model.r = pyo.Param(model.L)         # Resistance for branch l.
    model.x = pyo.Param(model.L)         # Reactance for branch l.
    model.b = pyo.Param(model.L)         # Shunt susceptance for branch l.
    model.p_node = pyo.Param(model.N)         # Real node at bus i.
    model.q_node = pyo.Param(model.N)         # Reactive load at bus i.
    model.i_max = pyo.Param(model.N, initialize=1.0e4) # Minimum voltage (p.u.)    
    model.vmin = pyo.Param(model.N, initialize=0.9) # Minimum voltage (p.u.)
    model.vmax = pyo.Param(model.N, initialize=1.1) # Maximum voltage (p.u.)
    # master_d is defined over LF: 1 if candidate is active, else 0.
    model.master_d = pyo.Param(model.LC, initialize=0)
    
    model.slack_v_sq = pyo.Param(initialize=1.0) # Slack bus voltage (p.u.)
    model.slack_node = pyo.Param(initialize=0)     # Slack bus index.
    
    model.M = pyo.Param(initialize=1e4)    # Big-M constant.
    
    return model