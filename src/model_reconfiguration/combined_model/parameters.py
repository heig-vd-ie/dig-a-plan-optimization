import pyomo.environ as pyo


def model_parameters(model: pyo.AbstractModel) -> pyo.AbstractModel:
    model.ρ = pyo.Param(mutable=True, default=1.0)  # ADMM penalty parameter
    model.γ_infeasibility = pyo.Param(default=1.0)
    model.γ_admm_penalty = pyo.Param(default=1.0)
    # ADMM params, now scenario‐indexed:
    model.zδ = pyo.Param(model.S, mutable=True, initialize=0.0)
    model.λδ = pyo.Param(model.S, mutable=True, initialize=0.0)
    model.zζ = pyo.Param(model.TrTaps, mutable=True, initialize=0.0)
    model.λζ = pyo.Param(model.TrTaps, mutable=True, initialize=0.0)
    model.voll = pyo.Param(mutable=True, default=1.0)  # Cost of load curtailment
    model.volp = pyo.Param(mutable=True, default=1.0)  # Cost of generation curtailment
    return model
