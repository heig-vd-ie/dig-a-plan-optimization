
module ScenariosGeneration

using Random

using ..Types

function random_partition(total::Float64, n::Int)
    breaks = sort([0.0; rand(n - 1) .* total; total])
    """
    Randomly split a total `total` into `n` non-negative values that sum to `total`
    """
    return [breaks[i + 1] - breaks[i] for i in 1:n]
end

# Generate random scenarios for each stage
function generate_scenarios(
    n_scenarios::Int,
    n_stages::Int,
    nodes::Vector{Types.Node};
    total_load_per_node::Float64 = 2.0,
    total_pv_per_node::Float64 = 1.0,
    total_budget::Float64 = 1000.0,
    seed_number::Int = 1234,
)
    Random.seed!(seed_number)
    Ω = Vector{Vector{Scenario}}(undef, n_scenarios)

    for s in 1:n_scenarios
        ω = Vector{Scenario}(undef, n_stages)

        # Randomly split total δ_load and δ_pv across stages, per node
        δ_load_splits = Dict{Types.Node, Vector{Float64}}()
        δ_pv_splits = Dict{Types.Node, Vector{Float64}}()

        for node in nodes
            δ_load_splits[node] = random_partition(total_load_per_node, n_stages)
            δ_pv_splits[node] = random_partition(total_pv_per_node, n_stages)
        end

        for t in 1:n_stages
            δ_load = Dict(node => δ_load_splits[node][t] for node in nodes)
            δ_pv = Dict(node => δ_pv_splits[node][t] for node in nodes)
            δ_b = rand(0.0:1.0:total_budget)  # or keep constant per scenario if needed

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
