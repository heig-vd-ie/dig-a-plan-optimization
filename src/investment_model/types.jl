module Types

export Grid, Scenario, PlanningParams

struct Grid
    nodes::Vector{Int64}
    edges::Vector{Tuple{Int64, Int64, Int64}}
    external_grid::Int64
    initial_capacity::Dict{Tuple{Int64, Int64, Int64}, Float64}
    load::Dict{Int64, Float64}
    pv::Dict{Int64, Float64}
    factor_pv::Dict{Tuple{Int64, Int64, Int64}, Dict{Int64, Float64}}
    factor_load::Dict{Tuple{Int64, Int64, Int64}, Dict{Int64, Float64}}
end

struct Scenario
    δ_load::Dict{Int64, Float64}
    δ_pv::Dict{Int64, Float64}
    δ_budget::Float64
end

struct PlanningParams
    grid::Grid
    n_stages::Int
    Ω::Vector{Vector{Scenario}}
    P::Vector{Float64}
    initial_budget::Float64
    investment_costs::Dict{Tuple{Int64, Int64, Int64}, Float64}
    penalty_costs_load::Dict{Int64, Float64}
    penalty_costs_pv::Dict{Int64, Float64}
    discount_rate::Float64
end


end # module Types 