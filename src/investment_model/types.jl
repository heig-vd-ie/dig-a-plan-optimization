module Types

export Grid, Scenario, PlanningParams

struct Grid
    nodes::Vector{String}
    edges::Vector{Tuple{String, String}}
    external_grid::String
    initial_capacity::Dict{Tuple{String, String}, Float64}
    load::Dict{String, Float64}
    pv::Dict{String, Float64}
    factor_pv::Dict{Tuple{String, String}, Dict{String, Float64}}
    factor_load::Dict{Tuple{String, String}, Dict{String, Float64}}
end

struct Scenario
    δ_load::Dict{String, Float64}
    δ_pv::Dict{String, Float64}
    δ_budget::Float64
end

struct PlanningParams
    grid::Grid
    n_stages::Int
    Ω::Vector{Vector{Scenario}}
    P::Vector{Float64}
    initial_budget::Float64
    investment_costs::Dict{Tuple{String, String}, Float64}
    penalty_costs_load::Dict{String, Float64}
    penalty_costs_pv::Dict{String, Float64}
    discount_rate::Float64
end


end # module Types 