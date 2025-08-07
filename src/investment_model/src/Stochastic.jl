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

    # Decision variables
    SDDP.parameterize(m, scenarios.Ω[stage], scenarios.P) do ω
        for node in grid.nodes
            JuMP.fix(vars.δ_load[node], ω.δ_load[node])
            JuMP.fix(vars.δ_pv[node], ω.δ_pv[node])
        end
        JuMP.fix(vars.δ_b, ω.δ_b)
        return nothing
    end

    # Expansion committed for next stage
    @constraint(m, [edge in grid.edges], states.δ_com[edge].out == vars.δ_cap[edge])
    # capacity update: add expansion from previous stage
    @constraint(
        m,
        [edge in grid.edges],
        states.cap[edge].out == states.cap[edge].in + states.δ_com[edge].in
    )
    @constraint(
        m,
        vars.investment_cost ==
        sum(params.investment_costs[edge] * vars.δ_cap[edge] for edge in grid.edges)
    )
    # Budget update
    @constraint(m, states.b_rem.out == states.b_rem.in - vars.investment_cost + vars.δ_b)

    # Calculate discount factor for NPV
    discount_factor = (1 / (1 + params.discount_rate))^(stage - 1)

    if stage == 1
        # Actual expansions (random variable minus unmet demand)
        @constraint(
            m,
            [node in grid.nodes],
            states.actual_load[node].out == states.actual_load[node].in
        )
        @constraint(
            m,
            [node in grid.nodes],
            states.actual_pv[node].out == states.actual_pv[node].in
        )
        # Unmet demand constraints
        @constraint(
            m,
            [node in grid.nodes],
            states.total_unmet_load[node].out == states.total_unmet_load[node].in
        )
        @constraint(
            m,
            [node in grid.nodes],
            states.total_unmet_pv[node].out == states.total_unmet_pv[node].in
        )
        @constraint(
            m,
            vars.obj ==
            discount_factor * (
                vars.investment_cost +
                sum(
                    params.penalty_costs_load[node] * states.total_unmet_load[node].out for
                    node in grid.nodes
                ) +
                sum(
                    params.penalty_costs_pv[node] * states.total_unmet_pv[node].out for
                    node in grid.nodes
                )
            )
        )
    else

        # Actual expansions (random variable minus unmet demand)
        @constraint(
            m,
            [node in grid.nodes],
            states.actual_load[node].out ==
            states.actual_load[node].in + vars.δ_load[node] - vars.unmet_load[node]
        )
        @constraint(
            m,
            [node in grid.nodes],
            states.actual_pv[node].out ==
            states.actual_pv[node].in + vars.δ_pv[node] - vars.unmet_pv[node]
        )

        # Unmet demand constraints
        @constraint(
            m,
            [node in grid.nodes],
            states.total_unmet_load[node].out ==
            states.total_unmet_load[node].in + vars.unmet_load[node]
        )
        @constraint(
            m,
            [node in grid.nodes],
            states.total_unmet_pv[node].out == states.total_unmet_pv[node].in + vars.unmet_pv[node]
        )

        # Flow conservation constraints
        for node in grid.nodes
            if node == grid.external_grid
                # External grid node: flow is equal to actual load minus actual external node flow
                @constraint(
                    m,
                    sum(
                        vars.flow[Edge(edge.id, edge.from, node.id)] for
                        edge in grid.edges if edge.to == node.id
                    ) - sum(
                        vars.flow[Edge(edge.id, node.id, edge.to)] for
                        edge in grid.edges if edge.from == node.id
                    ) == states.actual_load[node].out - vars.external_flow
                )
            else
                # Internal nodes: flow conservation with respect to actual load
                # and no external generation
                @constraint(
                    m,
                    sum(
                        vars.flow[Edge(edge.id, edge.from, node.id)] for
                        edge in grid.edges if edge.to == node.id
                    ) - sum(
                        vars.flow[Edge(edge.id, node.id, edge.to)] for
                        edge in grid.edges if edge.from == node.id
                    ) == states.actual_load[node].out
                )
            end
        end

        # Edge states.cap constraints: sum of actual expansions at connected nodes
        for edge in grid.edges
            @constraint(m, states.cap[edge].out >= vars.flow[edge])
            @constraint(m, states.cap[edge].out >= -vars.flow[edge])
            # Flow conservation for each edge
            @constraint(
                m,
                states.cap[edge].out >=
                sum(
                    states.actual_load[node].out * grid.factor_load[edge][node] for
                    node in grid.nodes
                ) +
                sum(states.actual_pv[node].out * grid.factor_pv[edge][node] for node in grid.nodes)
            )
        end

        # Objective: 
        @constraint(
            m,
            vars.obj ==
            discount_factor * (
                vars.investment_cost +
                sum(
                    params.penalty_costs_load[node] * states.total_unmet_load[node].out for
                    node in grid.nodes
                ) +
                sum(
                    params.penalty_costs_pv[node] * states.total_unmet_pv[node].out for
                    node in grid.nodes
                )
            )
        )
    end

    # Objective: investment + penalties for unmet demand, discounted to present value
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

end # module Stochastic
