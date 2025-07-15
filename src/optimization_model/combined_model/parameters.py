import pyomo.environ as pyo


def model_parameters(model: pyo.AbstractModel) -> pyo.AbstractModel:
    # Topology & radiality
    model.slack_node = pyo.Param()  # Slack bus index
    model.epsilon = pyo.Param(default=1)  # Small constant for radiality constraints
    # Line parameters
    model.r = pyo.Param(model.L)  # Resistance (pu)
    model.x = pyo.Param(model.L)  # Reactance (pu)
    model.b = pyo.Param(model.L)  # Shunt susceptance (pu)
    model.i_max = pyo.Param(model.L)  # Current limit (pu)
    # Candidate-to-actual transformer turns
    model.n_transfo = pyo.Param(model.C, default=1)
    # Nodal loads
    model.p_node = pyo.Param(model.N)  # Real load (pu)
    model.q_node = pyo.Param(model.N)  # Reactive load (pu)
    # Voltage limits
    model.v_min = pyo.Param(model.N, default=0.95)
    model.v_max = pyo.Param(model.N, default=1.05)
    # Slack bus voltage
    model.slack_node_v_sq = pyo.Param()
    # Big-M for switch on/off
    model.big_m = pyo.Param()

    return model
