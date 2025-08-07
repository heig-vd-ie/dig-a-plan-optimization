module Constraints

using SDDP, JuMP

using ..Types, ..Variables

export define_constraints!, define_first_stage_constraints!, define_subsequent_stage_constraints!

function define_constraints!(m::Model, grid::Types.Grid, vars, states, params::Types.PlanningParams)
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
    return nothing
end

function define_first_stage_constraints!(
    m::Model,
    grid::Types.Grid,
    states,
    vars,
    params::Types.PlanningParams,
    discount_factor,
)
    # Actual expansions (random variable minus unmet demand)
    @constraint(
        m,
        [node in grid.nodes],
        states.actual_load[node].out == states.actual_load[node].in
    )
    @constraint(m, [node in grid.nodes], states.actual_pv[node].out == states.actual_pv[node].in)
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
    # Objective: investment + penalties for unmet demand, discounted to present value
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
    return nothing
end

function define_subsequent_stage_constraints!(
    m::Model,
    grid::Types.Grid,
    states,
    vars,
    params::Types.PlanningParams,
    discount_factor,
)
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
                states.actual_load[node].out * grid.factor_load[edge][node] for node in grid.nodes
            ) + sum(states.actual_pv[node].out * grid.factor_pv[edge][node] for node in grid.nodes)
        )
    end

    # Objective: investment + penalties for unmet demand, discounted to present value
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
    return nothing
end
end