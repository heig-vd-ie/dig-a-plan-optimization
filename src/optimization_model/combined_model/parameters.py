import pyomo.environ as pyo


def model_parameters(model: pyo.AbstractModel) -> pyo.AbstractModel:
    # Topology & radiality
    model.slack_node = pyo.Param()  # Slack bus index
    model.small_m = pyo.Param()  # Small constant for radiality constraints
    model.rho = pyo.Param(default=1.0)  # ADMM penalty parameter
    # Line parameters
    model.r = pyo.Param(model.L)  # Resistance (pu)
    model.x = pyo.Param(model.L)  # Reactance (pu)
    model.b = pyo.Param(model.L)  # Shunt susceptance (pu)
    model.i_max = pyo.Param(model.L)  # Current limit (pu)
    # Candidate-to-actual transformer turns
    model.n_transfo = pyo.Param(model.C)
    # Nodal loads
    model.p_node = pyo.Param(model.SCEN, model.N, mutable=True)  # Real load (pu)
    model.q_node = pyo.Param(model.SCEN, model.N, mutable=True)  # Reactive load (pu)
    # Voltage limits
    model.v_min = pyo.Param(model.N, default=0.95)  # Minimum voltage (pu)
    model.v_max = pyo.Param(model.N, default=1.05)  # Maximum voltage (pu)
    # Slack bus voltage
    model.slack_node_v_sq = pyo.Param(default=1.0)  # Slack bus voltage squared (pu)
    # Big-M for switch on/off
    model.big_m = pyo.Param()
    model.weight_infeasibility = pyo.Param(default=1.0)
    model.weight_penalty = pyo.Param(default=1e-6)

    # ADMM params, now scenario‐indexed:
    model.del_param = pyo.Param(model.SCEN, model.S, mutable=True, initialize=0.0)
    model.u_param = pyo.Param(model.SCEN, model.S, mutable=True, initialize=0.0)

    return model
