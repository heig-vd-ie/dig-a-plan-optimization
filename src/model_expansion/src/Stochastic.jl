module Stochastic

using SDDP, JuMP, HiGHS, Random
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
    if stage == 1
        define_first_stage_constraints!(m, grid, states)
    else
        define_subsequent_stage_constraints!(m, grid, params, states, vars)
    end
    define_objective!(m, grid, vars, states, params, stage)
    @stageobjective(m, vars.obj)

    return m
end

function stochastic_planning(
    grid::Grid,
    scenarios::Scenarios,
    params::PlanningParams,
    iteration_limit::Int64,
    n_simulations::Int64,
    risk_measure::SDDP.AbstractRiskMeasure,
    seed::Int = 1234,
)
    Random.seed!(seed)

    model = SDDP.LinearPolicyGraph(;
        stages = params.n_stages,
        sense = :Min,
        lower_bound = 0.0,
        optimizer = HiGHS.Optimizer,
    ) do m, stage
        model_builder(m, grid, stage, scenarios, params)
    end

    SDDP.train(model, risk_measure = risk_measure, iteration_limit = iteration_limit)

    Random.seed!(seed)

    simulations = SDDP.simulate(
        model,
        n_simulations,
        [
            :investment_cost,
            :total_unmet_load,
            :total_unmet_pv,
            :cap,
            :δ_cap,
            :obj,
            :δ_load,
            :δ_pv,
            :δ_b,
        ],
    )
    objectives = [sum(stage[:obj] for stage in data) for data in simulations]

    return simulations, objectives
end

end
