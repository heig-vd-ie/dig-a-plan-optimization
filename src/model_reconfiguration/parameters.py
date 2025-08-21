import pyomo.environ as pyo


def model_parameters(model: pyo.AbstractModel) -> pyo.AbstractModel:
    # Topology & radiality
    model.number_of_lines = pyo.Param(within=pyo.NonNegativeIntegers)
    model.ε = pyo.Param(
        within=pyo.NonNegativeReals
    )  # Small constant for radiality constraints
    model.γ_trafo_loss = pyo.Param(
        default=1.0, within=pyo.NonNegativeReals
    )  # Trafo loss penalty
    # Line parameters
    model.r = pyo.Param(model.E, within=pyo.NonNegativeReals)  # Resistance (pu)
    model.x = pyo.Param(model.E, within=pyo.NonNegativeReals)  # Reactance (pu)
    model.b = pyo.Param(model.E, within=pyo.NonNegativeReals)  # Shunt susceptance (pu)
    model.i_max = pyo.Param(model.E, within=pyo.NonNegativeReals)  # Current limit (pu)
    # Nodal loads
    model.p_node_cons = pyo.Param(
        model.NΩ, within=pyo.NonNegativeReals
    )  # Real load (pu)
    model.q_node_cons = pyo.Param(
        model.NΩ, within=pyo.NonNegativeReals
    )  # Reactive load (pu)
    model.p_node_prod = pyo.Param(
        model.NΩ, within=pyo.NonNegativeReals
    )  # Real production (pu)
    model.q_node_prod = pyo.Param(
        model.NΩ, within=pyo.NonNegativeReals
    )  # Reactive production (pu)
    # Voltage limits
    model.v_min = pyo.Param(
        model.N, default=0.95, within=pyo.NonNegativeReals
    )  # Minimum voltage (pu)
    model.v_max = pyo.Param(
        model.N, default=1.05, within=pyo.NonNegativeReals
    )  # Maximum voltage (pu)
    # Slack bus voltage
    model.slack_node_v_sq = pyo.Param(model.Ω, default=1.0, within=pyo.NonNegativeReals)
    # Big-M for switch on/off
    model.big_m = pyo.Param(within=pyo.NonNegativeReals)

    model.node_cons_installed_param = pyo.Param(model.N, within=pyo.NonNegativeReals)
    model.node_prod_installed_param = pyo.Param(model.N, within=pyo.NonNegativeReals)

    return model
