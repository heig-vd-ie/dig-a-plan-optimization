module Stochastic

using Pkg

Pkg.activate(@__DIR__)     # Activates the environment in the parent folder
Pkg.instantiate()                   # Installs all packages if not already installed

using SDDP, JuMP, HiGHS
using ..Types

export stochastic_planning

function subproblem_builder(subproblem::Model, stage::Int, params::Types.PlanningParams)
    grid = params.grid
    # State variables
    @variable(subproblem, budget_remaining >= 0, SDDP.State, initial_value = params.initial_budget)
    @variable(subproblem, capacity[edge in grid.edges] >= 0, SDDP.State, initial_value = grid.initial_capacity[edge])
    # Decision variables
    @variable(subproblem, δ_capacity[edge in grid.edges] >= 0)  # expansion decision
    @variable(subproblem, expansion_committed[edge in grid.edges] >= 0, SDDP.State, initial_value = 0.0)
    @variable(subproblem, investment_cost >= 0)
    # Random variables (fixed by scenario)
    @variable(subproblem, δ_load[node in grid.nodes])
    @variable(subproblem, δ_pv[node in grid.nodes])
    @variable(subproblem, δ_budget)

    SDDP.parameterize(subproblem, params.Ω[stage], params.P) do ω
        for node in grid.nodes
            JuMP.fix(δ_load[node], ω.δ_load[node])
            JuMP.fix(δ_pv[node], ω.δ_pv[node])
        end
        JuMP.fix(δ_budget, ω.δ_budget)
        return nothing
    end

    # Unmet demand variables per node and technology
    @variable(subproblem, total_unmet_load[node in grid.nodes] >= 0, SDDP.State, initial_value = 0.0)
    @variable(subproblem, total_unmet_pv[node in grid.nodes] >= 0, SDDP.State, initial_value = 0.0)

    @variable(subproblem, actual_load[node in grid.nodes] >= 0, SDDP.State, initial_value = params.grid.load[node])
    @variable(subproblem, actual_pv[node in grid.nodes] >= 0, SDDP.State, initial_value = params.grid.pv[node])

    @variable(subproblem, unmet_load[node in grid.nodes])
    @variable(subproblem, unmet_pv[node in grid.nodes])

    @variable(subproblem, flow[edge in grid.edges])

    @variable(subproblem, external_flow)
    @variable(subproblem, objective_value)

    # Expansion committed for next stage
    @constraint(subproblem, [edge in grid.edges], expansion_committed[edge].out == δ_capacity[edge])
    # Capacity update: add expansion from previous stage
    @constraint(subproblem, [edge in grid.edges], capacity[edge].out == capacity[edge].in + expansion_committed[edge].in)
    @constraint(subproblem, investment_cost == sum(params.investment_costs[edge] * δ_capacity[edge] for edge in grid.edges))
    # Budget update
    @constraint(subproblem, budget_remaining.out == budget_remaining.in - investment_cost + δ_budget)

    # Calculate discount factor for NPV
    discount_factor = (1 / (1 + params.discount_rate)) ^ (stage - 1)

    if stage == 1
        # Actual expansions (random variable minus unmet demand)
        @constraint(subproblem, [node in grid.nodes], actual_load[node].out == actual_load[node].in)
        @constraint(subproblem, [node in grid.nodes], actual_pv[node].out == actual_pv[node].in)
        # Unmet demand constraints
        @constraint(subproblem, [node in grid.nodes], total_unmet_load[node].out == total_unmet_load[node].in )
        @constraint(subproblem, [node in grid.nodes], total_unmet_pv[node].out == total_unmet_pv[node].in)
        @constraint(subproblem, objective_value == discount_factor * (
            investment_cost +
            sum(params.penalty_costs_load[node] * total_unmet_load[node].out for node in grid.nodes) +
            sum(params.penalty_costs_pv[node] * total_unmet_pv[node].out for node in grid.nodes)
        ))
    else

        # Actual expansions (random variable minus unmet demand)
        @constraint(subproblem, [node in grid.nodes], actual_load[node].out == actual_load[node].in + δ_load[node] - unmet_load[node])
        @constraint(subproblem, [node in grid.nodes], actual_pv[node].out == actual_pv[node].in + δ_pv[node] - unmet_pv[node])

        # Unmet demand constraints
        @constraint(subproblem, [node in grid.nodes], total_unmet_load[node].out == total_unmet_load[node].in + unmet_load[node])
        @constraint(subproblem, [node in grid.nodes], total_unmet_pv[node].out == total_unmet_pv[node].in + unmet_pv[node])


        # Flow conservation constraints
        for node in grid.nodes
            if node == grid.external_grid
                # External grid node: flow is equal to actual load minus actual external node flow
                @constraint(subproblem, sum(flow[(edge.id, edge.from, node)] for edge in grid.edges if edge.to == node) -
                            sum(flow[(edge.id, node, edge.to)] for edge in grid.edges if edge.from == node) ==
                            actual_load[node].out - external_flow)
            else
                # Internal nodes: flow conservation with respect to actual load
                # and no external generation
                @constraint(subproblem, sum(flow[(edge.id, edge.from, node)] for edge in grid.edges if edge.to == node) -
                            sum(flow[(edge.id, node, edge.to)] for edge in grid.edges if edge.from == node) == actual_load[node].out)
            end
        end

        # Edge capacity constraints: sum of actual expansions at connected nodes
        for edge in grid.edges
            @constraint(subproblem, capacity[edge].out >= flow[edge])
            @constraint(subproblem, capacity[edge].out >= -flow[edge])
            # Flow conservation for each edge
            @constraint(subproblem,
                capacity[edge].out >= sum(actual_load[node].out * params.grid.factor_load[edge][node] for node in grid.nodes) +
                                sum(actual_pv[node].out * params.grid.factor_pv[edge][node] for node in grid.nodes)
            )
        end


        # Objective: 
        @constraint(subproblem, objective_value == discount_factor * (
            investment_cost +
            sum(params.penalty_costs_load[node] * total_unmet_load[node].out for node in grid.nodes) +
            sum(params.penalty_costs_pv[node] * total_unmet_pv[node].out for node in grid.nodes)
        ))
    end

    # Objective: investment + penalties for unmet demand, discounted to present value
    @stageobjective(subproblem, objective_value)

    return subproblem
end

function stochastic_planning(params::Types.PlanningParams)
    model = SDDP.LinearPolicyGraph(;
        stages = params.n_stages,
        sense = :Min,
        lower_bound=0.0,
        optimizer=HiGHS.Optimizer,
    ) do subproblem, stage
        subproblem_builder(subproblem, stage, params)
    end
    return model
end

end # module Stochastic