import pyomo.environ as pyo

def slave_model_parameters(model: pyo.AbstractModel) -> pyo.AbstractModel:
    model.r = pyo.Param(model.L)         # Resistance in pu for branch l.
    model.x = pyo.Param(model.L)         # Reactance in pu for branch l.
    model.b = pyo.Param(model.L)         # Shunt susceptance in pu for branch l.
    model.n_transfo = pyo.Param(model.LC, default=1)  # Transformer turn ration in pu for branch l. 
    model.p_node = pyo.Param(model.N)         # Real node at bus i.
    model.q_node = pyo.Param(model.N)         # Reactive load at bus i.
    model.i_max = pyo.Param(model.L, default=1.0e4) # Minimum voltage (p.u.)    
    model.vmin = pyo.Param(model.N, default=0.9) # Minimum voltage (p.u.)
    model.vmax = pyo.Param(model.N, default=1.1) # Maximum voltage (p.u.)
    # master_d is defined over LF: 1 if candidate is active, else 0.
    model.master_d = pyo.Param(model.LC, default= 0, mutable=True)
    
    model.slack_node_v_sq = pyo.Param(default=1.0) # Slack bus voltage (p.u.)
    model.slack_node = pyo.Param()     # Slack bus index.
    
    model.M = pyo.Param(default=1e4)    # Big-M constant.
    model.v_penalty_factor = pyo.Param(default=1e-3)
    model.i_penalty_factor  = pyo.Param(default=1e-3)
    
    return model