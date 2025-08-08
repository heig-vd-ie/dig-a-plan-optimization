
module Wasserstein
using HiGHS, SDDP
using ..Types
function wasserstein_norm(x::SDDP.Noise{Types.Scenario}, y::SDDP.Noise{Types.Scenario})
    s1, s2 = x.term, y.term
    # Compute Euclidean distance over all numeric fields
    delta_load_diff = sum(abs(s1.δ_load[n] - s2.δ_load[n]) for n in keys(s1.δ_load))
    delta_pv_diff = sum(abs(s1.δ_pv[n] - s2.δ_pv[n]) for n in keys(s1.δ_pv))
    delta_budget_diff = abs(s1.δ_b - s2.δ_b)
    return delta_load_diff + delta_pv_diff + delta_budget_diff
end

function risk_measure(α::Real)
    return SDDP.Wasserstein(wasserstein_norm, HiGHS.Optimizer; alpha = α)
end
end