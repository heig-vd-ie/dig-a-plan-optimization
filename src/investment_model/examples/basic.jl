# push!(LOAD_PATH, joinpath(@__DIR__, ".."))
using InvestmentModel
using SDDP
using Random

using ..Types, ..Stochastic

# === Utility functions (inlined) ===
"""
Randomly split a total `total` into `n` non-negative values that sum to `total`
"""
function random_partition(total::Float64, n::Int)
    breaks = sort([0.0; rand(n - 1) .* total; total])
    return [breaks[i + 1] - breaks[i] for i in 1:n]
end

# Generate random scenarios for each stage
function generate_scenarios(
    n_scenarios::Int,
    n_stages::Int,
    nodes::Vector{Types.Node};
    total_load_per_node::Float64 = 2.0,
    total_pv_per_node::Float64 = 1.0,
)
    scenarios = Vector{Vector{Scenario}}(undef, n_scenarios)

    for s in 1:n_scenarios
        scenario_path = Vector{Scenario}(undef, n_stages)

        # Randomly split total delta_load and delta_pv across stages, per node
        delta_load_splits = Dict{Types.Node, Vector{Float64}}()
        delta_pv_splits = Dict{Types.Node, Vector{Float64}}()

        for node in nodes
            delta_load_splits[node] = random_partition(total_load_per_node, n_stages)
            delta_pv_splits[node] = random_partition(total_pv_per_node, n_stages)
        end

        for t in 1:n_stages
            delta_load = Dict(node => delta_load_splits[node][t] for node in nodes)
            delta_pv = Dict(node => delta_pv_splits[node][t] for node in nodes)
            delta_budget = rand(0.0:1.0:1000.0)  # or keep constant per scenario if needed

            scenario_path[t] = Scenario(delta_load, delta_pv, delta_budget)
        end

        scenarios[s] = scenario_path
    end

    return [[scenarios[j][i] for j in 1:length(scenarios)] for i in 1:length(scenarios[1])]  # Returns Vector of scenario paths, each path is a Vector of Scenario
end

# Generate cost dictionaries for grid sections/edges
function generate_costs(edges, nodes)
    investment_costs = Dict(e => rand(90.0:95:100.0) for e in edges)
    penalty_costs_load = Dict(n => 6000.0 for n in nodes)
    penalty_costs_pv = Dict(n => 6000.0 for n in nodes)
    return investment_costs, penalty_costs_load, penalty_costs_pv
end

function generate_factor_load(edges, nodes)
    return Dict(edge => Dict(node => rand(0.0:0.1:0.2) for node in nodes) for edge in edges)
end

function generate_factor_pv(edges, nodes)
    return Dict(edge => Dict(node => rand(0.0:0.1:0.2) for node in nodes) for edge in edges)
end

# === Main example ===

Random.seed!(1234)
n_stages = 5
n_scenarios = 100
n_iterations = 10
n_simulations = 1000
nodes = [Types.Node(1), Types.Node(2), Types.Node(3), Types.Node(4)]
edges = [
    Types.Edge(1, 1, 2),
    Types.Edge(2, 2, 3),
    Types.Edge(3, 3, 1),
    Types.Edge(4, 2, 4),
    Types.Edge(5, 3, 4),
]
initial_cap = Dict(e => 1.0 for e in edges)
load = Dict(n => 1.0 for n in nodes)
pv = Dict(n => 0.1 for n in nodes)
factor_load = generate_factor_load(edges, nodes)
factor_pv = generate_factor_pv(edges, nodes)
grid = Types.Grid(nodes, edges, Types.Node(1), initial_cap, load, pv, factor_load, factor_pv)
Ω = generate_scenarios(n_scenarios, n_stages, nodes)
P = fill(1.0 / n_scenarios, n_scenarios)
investment_costs, penalty_costs_load, penalty_costs_pv = generate_costs(edges, nodes)
scenarios = Types.Scenarios(Ω, P)
params = Types.PlanningParams(
    n_stages,
    50.0,  # initial_budget
    investment_costs,
    penalty_costs_load,
    penalty_costs_pv,
    0.0,  # discount_rate
)
model1 = Stochastic.stochastic_planning(grid, scenarios, params)
model2 = Stochastic.stochastic_planning(grid, scenarios, params)

SDDP.train(model1, iteration_limit = n_iterations)

simulations1 = SDDP.simulate(
    model1,
    n_simulations,
    [:investment_cost, :total_unmet_load, :total_unmet_pv, :cap, :δ_cap, :obj],
)

# Compute and plot objective histogram for the last simulation
SDDP.train(model2, risk_measure = SDDP.Entropic(0.1), iteration_limit = n_iterations)

simulations2 = SDDP.simulate(
    model2,
    n_simulations,
    [:investment_cost, :total_unmet_load, :total_unmet_pv, :cap, :δ_cap, :obj],
)
objectives1 = [sum(stage[:obj] for stage in data) for data in simulations1]
objectives2 = [sum(stage[:obj] for stage in data) for data in simulations2]

using HiGHS
function wasserstein_norm(x::SDDP.Noise{Scenario}, y::SDDP.Noise{Scenario})
    s1, s2 = x.term, y.term
    # Compute Euclidean distance over all numeric fields
    delta_load_diff = sum(abs(s1.δ_load[n] - s2.δ_load[n]) for n in keys(s1.δ_load))
    delta_pv_diff = sum(abs(s1.δ_pv[n] - s2.δ_pv[n]) for n in keys(s1.δ_pv))
    delta_budget_diff = abs(s1.δ_b - s2.δ_b)
    return delta_load_diff + delta_pv_diff + delta_budget_diff
end

model3 = Stochastic.stochastic_planning(grid, scenarios, params)

SDDP.train(
    model3,
    risk_measure = SDDP.Wasserstein(wasserstein_norm, HiGHS.Optimizer; alpha = 1 / 20),
    iteration_limit = n_iterations,
)

simulations3 = SDDP.simulate(
    model3,
    n_simulations,
    [:investment_cost, :total_unmet_load, :total_unmet_pv, :cap, :δ_cap, :obj],
)
objectives3 = [sum(stage[:obj] for stage in data) for data in simulations3]

using Plots
histogram(
    [objectives1, objectives2, objectives3],
    normalize = true,
    nbins = 100,
    xlabel = "Objective Value",
    ylabel = "Frequency",
    title = "Objective Distribution Across Scenarios",
    label = ["Without risk measure" "With risk measure Χ" "Wasserstein"],
    opacity = 0.2,
    legend = :topright,
    fillalpha = 0.5,
    linecolor = [:blue :red :green],
    fillcolor = [:blue :red :green],
)
savefig(".cache/objective_histogram.pdf")

plt1 = SDDP.SpaghettiPlot(simulations1)

SDDP.add_spaghetti(plt1; title = "Total Unmet Load") do data
    return sum(data[:total_unmet_load][node].out for node in axes(data[:total_unmet_load], 1))
end
SDDP.add_spaghetti(plt1; title = "Total Unmet PV") do data
    return sum(data[:total_unmet_pv][node].out for node in axes(data[:total_unmet_pv], 1))
end
SDDP.add_spaghetti(plt1; title = "Capacity") do data
    return sum(data[:cap][node].out for node in axes(data[:cap], 1))
end
SDDP.plot(plt1, ".cache/example1.html", open = true)

plt2 = SDDP.SpaghettiPlot(simulations2)
SDDP.add_spaghetti(plt2; title = "Total Unmet Load") do data
    return sum(data[:total_unmet_load][node].out for node in axes(data[:total_unmet_load], 1))
end
SDDP.add_spaghetti(plt2; title = "Total Unmet PV") do data
    return sum(data[:total_unmet_pv][node].out for node in axes(data[:total_unmet_pv], 1))
end
SDDP.add_spaghetti(plt2; title = "Capacity") do data
    return sum(data[:cap][node].out for node in axes(data[:cap], 1))
end
SDDP.plot(plt2, ".cache/example2.html", open = true)

plt3 = SDDP.SpaghettiPlot(simulations3)
SDDP.add_spaghetti(plt3; title = "Total Unmet Load") do data
    return sum(data[:total_unmet_load][node].out for node in axes(data[:total_unmet_load], 1))
end
SDDP.add_spaghetti(plt3; title = "Total Unmet PV") do data
    return sum(data[:total_unmet_pv][node].out for node in axes(data[:total_unmet_pv], 1))
end
SDDP.add_spaghetti(plt3; title = "Capacity") do data
    return sum(data[:cap][node].out for node in axes(data[:cap], 1))
end
SDDP.plot(plt3, ".cache/example3.html", open = true)

import Plots
p1 = Vector{Plots.Plot}(undef, 3)
p2 = Vector{Plots.Plot}(undef, 3)
p3 = Vector{Plots.Plot}(undef, 3)
for (k, simulation) in enumerate([simulations1, simulations2, simulations3])
    p1[k] = SDDP.publication_plot(
        simulation;
        title = "Total Unmet Load",
        ylabel = "Unmet Load (MWh)",
        xlabel = "Stage",
    ) do data
        return sum(data[:total_unmet_load][node].out for node in axes(data[:total_unmet_load], 1))
    end
    p2[k] = SDDP.publication_plot(
        simulation;
        title = "Total Unmet PV",
        ylabel = "Unmet PV (MWh)",
        xlabel = "Stage",
    ) do data
        return sum(data[:total_unmet_pv][node].out for node in axes(data[:total_unmet_pv], 1))
    end
    p3[k] = SDDP.publication_plot(
        simulation;
        title = "Total Capacity",
        ylabel = "Capacity (MW)",
        xlabel = "Stage",
    ) do data
        return sum(data[:cap][node].out for node in axes(data[:cap], 1))
    end
end
plt = Plots.plot(
    p1[1],
    p2[1],
    p3[1],
    p1[2],
    p2[2],
    p3[2],
    p1[3],
    p2[3],
    p3[3],
    layout = (3, 3),
    size = (900, 1200),
)
Plots.savefig(plt, ".cache/stochastic_planning_results1.pdf")
