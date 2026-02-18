module Types

export Grid, Scenario, Scenarios, PlanningParams, Node, Edge

struct Node
    id::Int64
end

struct Edge
    id::Int64
    target::Int64
    source::Int64
end

struct Cut
    id::Int64
end

struct BenderCut
    θ::Float64
    λ_cap::Dict{Edge, Float64}
    λ_load::Dict{Node, Float64}
    λ_pv::Dict{Node, Float64}
    cap0::Dict{Edge, Float64}
    load0::Dict{Node, Float64}
    pv0::Dict{Node, Float64}
end

struct Grid
    nodes::Vector{Node}
    edges::Vector{Edge}
    cuts::Vector{Cut}
    external_grid::Node
    initial_cap::Dict{Edge, Float64}
    load::Dict{Node, Float64}
    pv::Dict{Node, Float64}
end

struct Scenario
    δ_load::Dict{Node, Float64}
    δ_pv::Dict{Node, Float64}
    δ_b::Float64
end

struct Scenarios
    Ω::Vector{Vector{Scenario}}
    P::Vector{Float64}
end

struct PlanningParams
    n_stages::Int
    initial_budget::Float64
    γ_cuts::Float64
    investment_costs::Dict{Edge, Float64}
    penalty_costs_load::Dict{Node, Float64}
    penalty_costs_pv::Dict{Node, Float64}
    penalty_costs_infeasibility::Float64
    discount_rate::Float64
    bender_cuts::Dict{Cut, BenderCut}
    years_per_stage::Int
    n_cut_scenarios::Int
    cut_slack_penalty::Float64
end

end
