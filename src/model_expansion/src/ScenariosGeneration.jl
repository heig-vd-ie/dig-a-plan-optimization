
module ScenariosGeneration

using Random, Distributions
export generate_scenarios

using ..Types

function random_partition(
    total::Float64,
    min_unit::Float64,
    n::Int,
    N_years::Int64,
    β::Float64 = 2.0,
)
    """
    Generate stochastic adoption curve using Weibull distribution.
    `n`          = number of stages
    `N_years`    = years per stage
    `total`      = total potential (kVA)
    `min_unit`   = unit size (kVA)
    `β`          = Weibull shape parameter
    """
    horizon = n * N_years                    # total simulation horizon (years)
    # Set scale λ so that F(horizon) = 0.5 → 50% adoption by end of horizon
    λ = horizon / (log(2))^(1 / β)
    dist = Weibull(β, λ)

    n_units = Int(round(total / min_unit))
    # Draw random adoption times within the horizon
    adoption_years = rand(dist, n_units)

    # Compute cumulative installed capacity by stage
    adopted =
        [min_unit * sum((a <= s * N_years for a in adoption_years); init = 0) for s in 1:n]

    # Clip to total potential (for numeric safety)
    adopted = [min(v, total) for v in adopted]

    return adopted
end

# Generate random scenarios for each stage
function generate_scenarios(
    n_scenarios::Int,
    n_stages::Int,
    nodes::Vector{Types.Node},
    load_potential::Dict{Types.Node, Float64},
    pv_potential::Dict{Types.Node, Float64},
    min_load::Float64 = 1.0,
    min_pv::Float64 = 5.0,
    yearly_budget::Float64 = 1000.0,
    N_years_per_stage::Int64 = 1,
    seed_number::Int = 1234,
)
    Random.seed!(seed_number)
    Ω = Vector{Vector{Scenario}}(undef, n_scenarios)

    for s in 1:n_scenarios
        ω = Vector{Scenario}(undef, n_stages)

        # Use Weibull distribution for each unit of load or PV to be added
        δ_load_splits = Dict{Types.Node, Vector{Float64}}()
        δ_pv_splits = Dict{Types.Node, Vector{Float64}}()

        for node in nodes
            δ_load_splits[node] =
                random_partition(load_potential[node], min_load, n_stages, N_years_per_stage)
            δ_pv_splits[node] =
                random_partition(pv_potential[node], min_pv, n_stages, N_years_per_stage)
        end

        for t in 1:n_stages
            δ_load = Dict(node => δ_load_splits[node][t] for node in nodes)
            δ_pv = Dict(node => δ_pv_splits[node][t] for node in nodes)
            δ_b = rand(0.0:1.0:yearly_budget)  # or keep constant per scenario if needed

            ω[t] = Scenario(δ_load, δ_pv, δ_b)
        end

        Ω[s] = ω
    end

    return [[Ω[j][i] for j in 1:length(Ω)] for i in 1:length(Ω[1])]  # Returns Vector of scenario paths, each path is a Vector of Scenario
end

# Generate cost dictionaries for grid sections/edges
function generate_costs(edges::Vector{Types.Edge}, nodes::Vector{Types.Node})
    investment_costs = Dict(e => rand(90.0:95:100.0) for e in edges)
    penalty_costs_load = Dict(n => 6000.0 for n in nodes)
    penalty_costs_pv = Dict(n => 6000.0 for n in nodes)
    penalty_costs_infeasibility = 6000
    return investment_costs, penalty_costs_load, penalty_costs_pv, penalty_costs_infeasibility
end

function generate_λ_load(cuts::Vector{Types.Cut}, nodes::Vector{Types.Node})
    return Dict(cut => Dict(node => rand(0.0:0.1:0.2) for node in nodes) for cut in cuts)
end

function generate_λ_pv(cuts::Vector{Types.Cut}, nodes::Vector{Types.Node})
    return Dict(cut => Dict(node => rand(0.0:0.1:0.2) for node in nodes) for cut in cuts)
end

function generate_λ_cap(cuts::Vector{Types.Cut}, edges::Vector{Types.Edge})
    return Dict(
        cut => Dict(edge => edge.id == cut.id ? -1.0 : 0.0 for edge in edges) for cut in cuts
    )
end

function generate_cap0(cuts::Vector{Types.Cut}, edges::Vector{Types.Edge})
    return Dict(cut => Dict(edge => 0.0 for edge in edges) for cut in cuts)
end

function generate_load0(cuts::Vector{Types.Cut}, nodes::Vector{Types.Node})
    return Dict(cut => Dict(node => 0.0 for node in nodes) for cut in cuts)
end

function generate_pv0(cuts::Vector{Types.Cut}, nodes::Vector{Types.Node})
    return Dict(cut => Dict(node => 0.0 for node in nodes) for cut in cuts)
end

function generate_θ(cuts::Vector{Types.Cut})
    return Dict(cut => 0.0 for cut in cuts)
end

end
