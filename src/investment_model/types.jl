module Types

export Grid, Scenario, PlanningParams, Node, Edge

struct Node
    id::Int64
end

struct Edge
    from::Int64
    to::Int64
    id::Int64
end

struct Grid
    nodes::Vector{Node}
    edges::Vector{Edge}
    external_grid::Node
    initial_capacity::Dict{Edge, Float64}
    load::Dict{Node, Float64}
    pv::Dict{Node, Float64}
    factor_pv::Dict{Edge, Dict{Node, Float64}}
    factor_load::Dict{Edge, Dict{Node, Float64}}
end

struct Scenario
    δ_load::Dict{Node, Float64}
    δ_pv::Dict{Node, Float64}
    δ_budget::Float64
end

struct PlanningParams
    n_stages::Int
    Ω::Vector{Vector{Scenario}}
    P::Vector{Float64}
    initial_budget::Float64
    investment_costs::Dict{Edge, Float64}
    penalty_costs_load::Dict{Node, Float64}
    penalty_costs_pv::Dict{Node, Float64}
    discount_rate::Float64
end

end # module Types 
