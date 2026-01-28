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
    out_of_sample_scenarios::Scenarios,
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

    print("Training started.")

    SDDP.train(model, risk_measure = risk_measure, iteration_limit = iteration_limit)

    println("Training has been completed.")

    in_sample_simulations, in_sample_objectives = in_sample_analysis(model, n_simulations, seed)

    println("In sample simulations generated.")

    out_of_sample_simulations, out_of_sample_objectives =
        out_of_sample_analysis(model, out_of_sample_scenarios, n_simulations, seed)

    println("Out sample simulations generated.")

    return in_sample_simulations,
    in_sample_objectives,
    out_of_sample_simulations,
    out_of_sample_objectives
end

function in_sample_analysis(model, n_simulations::Int64, seed::Int = 1234)
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

function out_of_sample_analysis(
    trained_model,
    out_of_sample_scenarios::Scenarios,
    n_simulations::Int64,
    seed::Int = 5678,
)
    Random.seed!(seed)

    # Create OutOfSampleMonteCarlo sampling scheme using your probabilities
    sampling_scheme =
        SDDP.OutOfSampleMonteCarlo(trained_model; use_insample_transition = true) do node
            stage = node  # For LinearPolicyGraph, node is just the stage number

            if stage <= length(out_of_sample_scenarios.Ω)
                stage_scenarios = out_of_sample_scenarios.Ω[stage]
                return [
                    SDDP.Noise(scenario, 1.0 / length(stage_scenarios)) for
                    scenario in stage_scenarios
                ]
            else
                return SDDP.Noise[]
            end
        end

    simulations = SDDP.simulate(
        trained_model,
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
        ];
        sampling_scheme = sampling_scheme,
    )

    objectives = [sum(stage[:obj] for stage in data) for data in simulations]
    return simulations, objectives
end

end
