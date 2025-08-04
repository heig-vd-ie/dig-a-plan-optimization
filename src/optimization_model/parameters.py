import pyomo.environ as pyo


def model_parameters(model: pyo.AbstractModel) -> pyo.AbstractModel:
    # Topology & radiality
    model.number_of_lines = pyo.Param()
    model.small_m = pyo.Param()  # Small constant for radiality constraints
    # Line parameters
    model.r = pyo.Param(model.L)  # Resistance (pu)
    model.x = pyo.Param(model.L)  # Reactance (pu)
    model.b = pyo.Param(model.L)  # Shunt susceptance (pu)
    model.i_max = pyo.Param(model.L)  # Current limit (pu)
    # Candidate-to-actual transformer turns
    model.n_transfo = pyo.Param(model.C)
    # Nodal loads
    model.p_node = pyo.Param(model.NΩ)  # Real load (pu)
    model.q_node = pyo.Param(model.NΩ)  # Reactive load (pu)
    # Voltage limits
    model.v_min = pyo.Param(model.N, default=0.95)  # Minimum voltage (pu)
    model.v_max = pyo.Param(model.N, default=1.05)  # Maximum voltage (pu)
    # Slack bus voltage
    model.slack_node_v_sq = pyo.Param(model.Ω, default=1.0)
    # Big-M for switch on/off
    model.big_m = pyo.Param()

    return model
