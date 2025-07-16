module Stochastic

using Pkg

Pkg.activate(@__DIR__)     # Activates the environment in the parent folder
Pkg.instantiate()                   # Installs all packages if not already installed

using SDDP, JuMP, HiGHS
using ..Types

export stochastic_planning

function subproblem_builder(sp::Model, stage::Int, params::Types.PlanningParams)
    grid = params.grid
    # State variables
    @variable(sp, budget_remaining >= 0, SDDP.State, initial_value = params.initial_budget)
    @variable(sp, capacity[edge in grid.edges] >= 0, SDDP.State, initial_value = grid.initial_capacity[edge])
    # Decision variables
    @variable(sp, δ_capacity[edge in grid.edges] >= 0)  # expansion decision
    @variable(sp, expansion_committed[edge in grid.edges] >= 0, SDDP.State, initial_value = 0.0)
    @variable(sp, investment_cost >= 0)
    # Random variables (fixed by scenario)
    @variable(sp, δ_load[node in grid.nodes])
    @variable(sp, δ_pv[node in grid.nodes])
    @variable(sp, δ_budget)

    SDDP.parameterize(sp, params.Ω[stage], params.P) do ω
        for node in grid.nodes
            JuMP.fix(δ_load[node], ω.δ_load[node])
            JuMP.fix(δ_pv[node], ω.δ_pv[node])
        end
        JuMP.fix(δ_budget, ω.δ_budget)
        return nothing
    end

    # Unmet demand variables per node and technology
    @variable(sp, total_unmet_load[node in grid.nodes] >= 0, SDDP.State, initial_value = 0.0)
    @variable(sp, total_unmet_pv[node in grid.nodes] >= 0, SDDP.State, initial_value = 0.0)

    @variable(sp, actual_load[node in grid.nodes] >= 0, SDDP.State, initial_value = params.grid.load[node])
    @variable(sp, actual_pv[node in grid.nodes] >= 0, SDDP.State, initial_value = params.grid.pv[node])

    @variable(sp, unmet_load[node in grid.nodes])
    @variable(sp, unmet_pv[node in grid.nodes])

    @variable(sp, flow[edge in grid.edges])

    @variable(sp, external_flow)
    @variable(sp, objective_value)

    # Expansion committed for next stage
    @constraint(sp, [edge in grid.edges], expansion_committed[edge].out == δ_capacity[edge])
    # Capacity update: add expansion from previous stage
    @constraint(sp, [edge in grid.edges], capacity[edge].out == capacity[edge].in + expansion_committed[edge].in)
    @constraint(sp, investment_cost == sum(params.investment_costs[edge] * δ_capacity[edge] for edge in grid.edges))
    # Budget update
    @constraint(sp, budget_remaining.out == budget_remaining.in - investment_cost + δ_budget)

    # Calculate discount factor for NPV
    discount_factor = (1 / (1 + params.discount_rate)) ^ (stage - 1)

    if stage == 1
        # Actual expansions (random variable minus unmet demand)
        @constraint(sp, [node in grid.nodes], actual_load[node].out == actual_load[node].in)
        @constraint(sp, [node in grid.nodes], actual_pv[node].out == actual_pv[node].in)
        # Unmet demand constraints
        @constraint(sp, [node in grid.nodes], total_unmet_load[node].out == total_unmet_load[node].in )
        @constraint(sp, [node in grid.nodes], total_unmet_pv[node].out == total_unmet_pv[node].in)
        @constraint(sp, objective_value == discount_factor * (
            investment_cost +
            sum(params.penalty_costs_load[node] * total_unmet_load[node].out for node in grid.nodes) +
            sum(params.penalty_costs_pv[node] * total_unmet_pv[node].out for node in grid.nodes)
        ))
    else

        # Actual expansions (random variable minus unmet demand)
        @constraint(sp, [node in grid.nodes], actual_load[node].out == actual_load[node].in + δ_load[node] - unmet_load[node])
        @constraint(sp, [node in grid.nodes], actual_pv[node].out == actual_pv[node].in + δ_pv[node] - unmet_pv[node])

        # Unmet demand constraints
        @constraint(sp, [node in grid.nodes], total_unmet_load[node].out == total_unmet_load[node].in + unmet_load[node])
        @constraint(sp, [node in grid.nodes], total_unmet_pv[node].out == total_unmet_pv[node].in + unmet_pv[node])


        # Flow conservation constraints
        for node in grid.nodes
            if node == grid.external_grid
                # External grid node: flow is equal to actual load minus actual external node flow
                @constraint(sp, sum(flow[(n1, node)] for (n1, n2) in grid.edges if n2 == node) -
                            sum(flow[(node, n2)] for (n1, n2) in grid.edges if n1 == node) ==
                            actual_load[node].out - external_flow)
            else
                # Internal nodes: flow conservation with respect to actual load
                # and no external generation
                @constraint(sp, sum(flow[(n1, node)] for (n1, n2) in grid.edges if n2 == node) -
                            sum(flow[(node, n2)] for (n1, n2) in grid.edges if n1 == node) == actual_load[node].out)
            end
        end

        # Edge capacity constraints: sum of actual expansions at connected nodes
        for (n1, n2) in grid.edges
            @constraint(sp,
                capacity[(n1, n2)].out >= flow[(n1, n2)]
            )
            @constraint(sp,
                capacity[(n1, n2)].out >= -flow[(n1, n2)]
            )
            # Flow conservation for each edge
            @constraint(sp,
                capacity[(n1, n2)].out >= sum(actual_load[node].out * params.grid.factor_load[(n1, n2)][node] for node in grid.nodes) +
                                sum(actual_pv[node].out * params.grid.factor_pv[(n1, n2)][node] for node in grid.nodes)
            )
        end


        # Objective: 
        @constraint(sp, objective_value == discount_factor * (
            investment_cost +
            sum(params.penalty_costs_load[node] * total_unmet_load[node].out for node in grid.nodes) +
            sum(params.penalty_costs_pv[node] * total_unmet_pv[node].out for node in grid.nodes)
        ))
    end

    # Objective: investment + penalties for unmet demand, discounted to present value
    @stageobjective(sp, objective_value)

    return sp
end

function stochastic_planning(params::Types.PlanningParams)
    model = SDDP.LinearPolicyGraph(;
        stages = params.n_stages,
        sense = :Min,
        lower_bound=0.0,
        optimizer=HiGHS.Optimizer,
    ) do sp, stage
        subproblem_builder(sp, stage, params)
    end
    return model
end

end # module Stochastic