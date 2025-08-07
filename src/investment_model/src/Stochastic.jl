module Stochastic

using SDDP, JuMP, HiGHS
export stochastic_planning

using ..Types, ..Variables, ..Constraints

function model_builder(
    m::Model,
    grid::Grid,
    stage::Int,
    scenarios::Scenarios,
    params::PlanningParams,
)
    states = define_state_variables!(m, params, grid)
    vars = define_decision_variables!(m, grid)

    SDDP.parameterize(m, scenarios.Ω[stage], scenarios.P) do ω
        for node in grid.nodes
            JuMP.fix(vars.δ_load[node], ω.δ_load[node])
            JuMP.fix(vars.δ_pv[node], ω.δ_pv[node])
        end
        JuMP.fix(vars.δ_b, ω.δ_b)
        return nothing
    end

    define_constraints!(m, grid, vars, states, params)
    discount_factor = (1 / (1 + params.discount_rate))^(stage - 1)
    if stage == 1
        define_first_stage_constraints!(m, grid, states, vars, params, discount_factor)
    else
        define_subsequent_stage_constraints!(m, grid, states, vars, params, discount_factor)
    end
    @stageobjective(m, vars.obj)

    return m
end

function stochastic_planning(grid::Grid, scenarios::Scenarios, params::PlanningParams)
    model = SDDP.LinearPolicyGraph(;
        stages = params.n_stages,
        sense = :Min,
        lower_bound = 0.0,
        optimizer = HiGHS.Optimizer,
    ) do m, stage
        model_builder(m, grid, stage, scenarios, params)
    end
    return model
end

end
