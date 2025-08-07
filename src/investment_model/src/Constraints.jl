module Constraints

using SDDP, JuMP

using ..Types, ..Variables

export define_constraints!,
    define_first_stage_constraints!, define_subsequent_stage_constraints!, define_objective!

function define_constraints!(
    m::Model,
    grid::Types.Grid,
    vars,
    states,
    params::Types.PlanningParams,
)
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
    return nothing
end

function define_first_stage_constraints!(m::Model, grid::Types.Grid, states)
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
    return nothing
end

function define_subsequent_stage_constraints!(m::Model, grid::Types.Grid, states, vars)
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

    # External grid node: flow is equal to actual load minus actual external node flow
    @constraint(
        m,
        [node in filter(n -> n == grid.external_grid, grid.nodes)],
        sum(
            vars.flow[Edge(edge.id, edge.from, node.id)] for
            edge in grid.edges if edge.to == node.id
        ) - sum(
            vars.flow[Edge(edge.id, node.id, edge.to)] for
            edge in grid.edges if edge.from == node.id
        ) == states.actual_load[node].out - vars.external_flow
    )
    # Internal nodes: flow conservation with respect to actual load
    # and no external generation
    @constraint(
        m,
        [node in filter(n -> n != grid.external_grid, grid.nodes)],
        sum(
            vars.flow[Edge(edge.id, edge.from, node.id)] for
            edge in grid.edges if edge.to == node.id
        ) - sum(
            vars.flow[Edge(edge.id, node.id, edge.to)] for
            edge in grid.edges if edge.from == node.id
        ) == states.actual_load[node].out
    )
    # Edge capacity constraints: sum of actual expansions at connected nodes
    @constraint(m, [edge in grid.edges], states.cap[edge].out >= vars.flow[edge])
    @constraint(m, [edge in grid.edges], states.cap[edge].out >= -vars.flow[edge])
    # Flow conservation for each edge
    @constraint(
        m,
        [edge in grid.edges],
        states.cap[edge].out >=
        sum(
            states.actual_load[node].out * grid.factor_load[edge][node] for node in grid.nodes
        ) + sum(states.actual_pv[node].out * grid.factor_pv[edge][node] for node in grid.nodes)
    )

    return nothing
end

function define_objective!(
    m::Model,
    grid::Types.Grid,
    vars,
    states,
    params::Types.PlanningParams,
    stage::Int,
)
    # Objective: investment + penalties for unmet demand, discounted to present value
    discount_factor = (1 / (1 + params.discount_rate))^(stage - 1)
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