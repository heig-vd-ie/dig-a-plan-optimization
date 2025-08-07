module Stochastic

using SDDP, JuMP, HiGHS
export stochastic_planning

using ..Types

function model_builder(
    m::Model,
    grid::Grid,
    stage::Int,
    scenarios::Scenarios,
    params::PlanningParams,
)
    m = model_variables(m, params)
    # State variables
    @variable(
        m,
        capacity[edge in grid.edges] >= 0,
        SDDP.State,
        initial_value = grid.initial_capacity[edge]
    )
    # Decision variables
    @variable(m, δ_capacity[edge in grid.edges] >= 0)  # expansion decision
    @variable(m, expansion_committed[edge in grid.edges] >= 0, SDDP.State, initial_value = 0.0)
    @variable(m, investment_cost >= 0)
    # Random variables (fixed by scenario)
    @variable(m, δ_load[node in grid.nodes])
    @variable(m, δ_pv[node in grid.nodes])
    @variable(m, δ_budget)

    SDDP.parameterize(m, scenarios.Ω[stage], scenarios.P) do ω
        for node in grid.nodes
            JuMP.fix(δ_load[node], ω.δ_load[node])
            JuMP.fix(δ_pv[node], ω.δ_pv[node])
        end
        JuMP.fix(δ_budget, ω.δ_budget)
        return nothing
    end

    # Unmet demand variables per node and technology
    @variable(m, total_unmet_load[node in grid.nodes] >= 0, SDDP.State, initial_value = 0.0)
    @variable(m, total_unmet_pv[node in grid.nodes] >= 0, SDDP.State, initial_value = 0.0)

    @variable(m, actual_load[node in grid.nodes] >= 0, SDDP.State, initial_value = grid.load[node])
    @variable(m, actual_pv[node in grid.nodes] >= 0, SDDP.State, initial_value = grid.pv[node])

    @variable(m, unmet_load[node in grid.nodes])
    @variable(m, unmet_pv[node in grid.nodes])

    @variable(m, flow[edge in grid.edges])

    @variable(m, external_flow)
    @variable(m, objective_value)

    # Expansion committed for next stage
    @constraint(m, [edge in grid.edges], expansion_committed[edge].out == δ_capacity[edge])
    # Capacity update: add expansion from previous stage
    @constraint(
        m,
        [edge in grid.edges],
        capacity[edge].out == capacity[edge].in + expansion_committed[edge].in
    )
    @constraint(
        m,
        investment_cost ==
        sum(params.investment_costs[edge] * δ_capacity[edge] for edge in grid.edges)
    )
    # Budget update
    @constraint(m, budget_remaining.out == budget_remaining.in - investment_cost + δ_budget)

    # Calculate discount factor for NPV
    discount_factor = (1 / (1 + params.discount_rate))^(stage - 1)

    if stage == 1
        # Actual expansions (random variable minus unmet demand)
        @constraint(m, [node in grid.nodes], actual_load[node].out == actual_load[node].in)
        @constraint(m, [node in grid.nodes], actual_pv[node].out == actual_pv[node].in)
        # Unmet demand constraints
        @constraint(
            m,
            [node in grid.nodes],
            total_unmet_load[node].out == total_unmet_load[node].in
        )
        @constraint(m, [node in grid.nodes], total_unmet_pv[node].out == total_unmet_pv[node].in)
        @constraint(
            m,
            objective_value ==
            discount_factor * (
                investment_cost +
                sum(
                    params.penalty_costs_load[node] * total_unmet_load[node].out for
                    node in grid.nodes
                ) +
                sum(params.penalty_costs_pv[node] * total_unmet_pv[node].out for node in grid.nodes)
            )
        )
    else

        # Actual expansions (random variable minus unmet demand)
        @constraint(
            m,
            [node in grid.nodes],
            actual_load[node].out == actual_load[node].in + δ_load[node] - unmet_load[node]
        )
        @constraint(
            m,
            [node in grid.nodes],
            actual_pv[node].out == actual_pv[node].in + δ_pv[node] - unmet_pv[node]
        )

        # Unmet demand constraints
        @constraint(
            m,
            [node in grid.nodes],
            total_unmet_load[node].out == total_unmet_load[node].in + unmet_load[node]
        )
        @constraint(
            m,
            [node in grid.nodes],
            total_unmet_pv[node].out == total_unmet_pv[node].in + unmet_pv[node]
        )

        # Flow conservation constraints
        for node in grid.nodes
            if node == grid.external_grid
                # External grid node: flow is equal to actual load minus actual external node flow
                @constraint(
                    m,
                    sum(
                        flow[(edge.id, edge.from, node)] for edge in grid.edges if edge.to == node
                    ) - sum(
                        flow[(edge.id, node, edge.to)] for edge in grid.edges if edge.from == node
                    ) == actual_load[node].out - external_flow
                )
            else
                # Internal nodes: flow conservation with respect to actual load
                # and no external generation
                @constraint(
                    m,
                    sum(
                        flow[(edge.id, edge.from, node)] for edge in grid.edges if edge.to == node
                    ) - sum(
                        flow[(edge.id, node, edge.to)] for edge in grid.edges if edge.from == node
                    ) == actual_load[node].out
                )
            end
        end

        # Edge capacity constraints: sum of actual expansions at connected nodes
        for edge in grid.edges
            @constraint(m, capacity[edge].out >= flow[edge])
            @constraint(m, capacity[edge].out >= -flow[edge])
            # Flow conservation for each edge
            @constraint(
                m,
                capacity[edge].out >=
                sum(actual_load[node].out * grid.factor_load[edge][node] for node in grid.nodes) +
                sum(actual_pv[node].out * grid.factor_pv[edge][node] for node in grid.nodes)
            )
        end

        # Objective: 
        @constraint(
            m,
            objective_value ==
            discount_factor * (
                investment_cost +
                sum(
                    params.penalty_costs_load[node] * total_unmet_load[node].out for
                    node in grid.nodes
                ) +
                sum(params.penalty_costs_pv[node] * total_unmet_pv[node].out for node in grid.nodes)
            )
        )
    end

    # Objective: investment + penalties for unmet demand, discounted to present value
    @stageobjective(m, objective_value)

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

end # module Stochastic
