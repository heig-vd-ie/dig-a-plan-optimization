import pyomo.environ as pyo


def model_parameters(model: pyo.AbstractModel) -> pyo.AbstractModel:
    model.ρ = pyo.Param(mutable=True, default=1.0)  # ADMM penalty parameter
    model.γ_infeasibility = pyo.Param(default=1.0)
    model.γ_penalty = pyo.Param(default=1e-6)
    model.γ_admm_penalty = pyo.Param(default=1.0)
    # ADMM params, now scenario‐indexed:
    model.z = pyo.Param(model.S, mutable=True, initialize=0.0)
    model.λ = pyo.Param(model.S, mutable=True, initialize=0.0)
    return model
