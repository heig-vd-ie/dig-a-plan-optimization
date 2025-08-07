
module Variables
using SDDP, JuMP, HiGHS

using ..Types

export model_variables

function model_variables(m::Model, params::Types.PlanningParams)
    # State variables
    @variable(m, budget_remaining >= 0, SDDP.State, initial_value = params.initial_budget)

    return m
end
end