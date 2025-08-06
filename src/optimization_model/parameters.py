import pyomo.environ as pyo


def model_parameters(model: pyo.AbstractModel) -> pyo.AbstractModel:
    # Topology & radiality
    model.number_of_lines = pyo.Param()
    model.ε = pyo.Param()  # Small constant for radiality constraints
    # Line parameters
    model.r = pyo.Param(model.E)  # Resistance (pu)
    model.x = pyo.Param(model.E)  # Reactance (pu)
    model.b = pyo.Param(model.E)  # Shunt susceptance (pu)
    model.i_max = pyo.Param(model.E)  # Current limit (pu)
    # Candidate-to-actual transformer turns
    model.n_transfo = pyo.Param(model.C)
    # Nodal loads
    model.p_node_cons = pyo.Param(model.NΩ)  # Real load (pu)
    model.q_node_cons = pyo.Param(model.NΩ)  # Reactive load (pu)
    model.p_node_prod = pyo.Param(model.NΩ)  # Real production (pu)
    model.q_node_prod = pyo.Param(model.NΩ)
    # Voltage limits
    model.v_min = pyo.Param(model.N, default=0.95)  # Minimum voltage (pu)
    model.v_max = pyo.Param(model.N, default=1.05)  # Maximum voltage (pu)
    # Slack bus voltage
    model.slack_node_v_sq = pyo.Param(model.Ω, default=1.0)
    # Big-M for switch on/off
    model.big_m = pyo.Param()

    return model
