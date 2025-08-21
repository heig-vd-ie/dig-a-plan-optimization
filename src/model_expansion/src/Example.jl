using ExpansionModel
using SDDP
using Plots
import Plots
using JSON

using ..Types, ..Stochastic, ..ScenariosGeneration, ..Wasserstein

# Function to convert Node objects to their IDs in nested structures
function serialize_nodes(obj)
    if isa(obj, Types.Node)
        return obj.id
    elseif isa(obj, Dict)
        # Handle dictionaries with Node keys specially
        new_dict = Dict()
        for (k, v) in obj
            new_key = isa(k, Types.Node) ? string(k.id) : serialize_nodes(k)
            new_dict[new_key] = serialize_nodes(v)
        end
        return new_dict
    elseif isa(obj, Array)
        return [serialize_nodes(item) for item in obj]
    else
        return obj
    end
end

# === Main example ===
n_stages = 10
n_scenarios = 100
iteration_limit = 10
n_simulations = 5000
nodes = [Types.Node(1), Types.Node(2), Types.Node(3), Types.Node(4)]
edges = [
    Types.Edge(1, 1, 2),
    Types.Edge(2, 2, 3),
    Types.Edge(3, 3, 1),
    Types.Edge(4, 2, 4),
    Types.Edge(5, 3, 4),
]
cuts = [Types.Cut(1), Types.Cut(2), Types.Cut(3), Types.Cut(4)]
initial_cap = Dict(e => 1.0 for e in edges)
load = Dict(n => 1.0 for n in nodes)
pv = Dict(n => 0.1 for n in nodes)
grid = Types.Grid(nodes, edges, cuts, Types.Node(1), initial_cap, load, pv)
Ω = ScenariosGeneration.generate_scenarios(n_scenarios, n_stages, nodes)
P = fill(1.0 / n_scenarios, n_scenarios)

# Export scenarios to JSON
scenarios_data_raw = Dict("Ω" => Ω, "P" => P)

# Convert to JSON string, then fix Node representations
json_string = JSON.json(scenarios_data_raw, 2)
# Replace "Node(X)" with just "X"
json_string = replace(json_string, r"\"Node\((\d+)\)\"" => s"\"\1\"")

isdir(".cache") || mkpath(".cache")
open(".cache/scenarios.json", "w") do file
    write(file, json_string)
end
println("Scenarios exported to .cache/scenarios.json")

investment_costs, penalty_costs_load, penalty_costs_pv =
    ScenariosGeneration.generate_costs(edges, nodes)
scenarios = Types.Scenarios(Ω, P)
λ_cap = ScenariosGeneration.generate_λ_cap(cuts, edges)
λ_load = ScenariosGeneration.generate_λ_load(cuts, nodes)
λ_pv = ScenariosGeneration.generate_λ_pv(cuts, nodes)
cap0 = ScenariosGeneration.generate_cap0(cuts, edges)
load0 = ScenariosGeneration.generate_load0(cuts, nodes)
pv0 = ScenariosGeneration.generate_pv0(cuts, nodes)
θ = ScenariosGeneration.generate_θ(cuts)
bender_cuts = Dict(
    cut => Types.BenderCut(
        θ[cut],
        λ_cap[cut],
        λ_load[cut],
        λ_pv[cut],
        cap0[cut],
        load0[cut],
        pv0[cut],
    ) for cut in cuts
)

# Export bender cuts to JSON
bender_cuts_data = Dict(
    "cuts" => Dict(
        cut.id => Dict(
            "θ" => bender_cuts[cut].θ,
            "λ_cap" => Dict(edge.id => bender_cuts[cut].λ_cap[edge] for edge in edges),
            "λ_load" => Dict(node.id => bender_cuts[cut].λ_load[node] for node in nodes),
            "λ_pv" => Dict(node.id => bender_cuts[cut].λ_pv[node] for node in nodes),
            "cap0" => Dict(edge.id => bender_cuts[cut].cap0[edge] for edge in edges),
            "load0" => Dict(node.id => bender_cuts[cut].load0[node] for node in nodes),
            "pv0" => Dict(node.id => bender_cuts[cut].pv0[node] for node in nodes),
        ) for cut in cuts
    ),
)

open(".cache/bender_cuts.json", "w") do file
    JSON.print(file, bender_cuts_data, 2)
end
println("Bender cuts exported to .cache/bender_cuts.json")

params = Types.PlanningParams(
    n_stages,
    50.0,  # initial_budget
    0.0,  # γ_cuts
    investment_costs,
    penalty_costs_load,
    penalty_costs_pv,
    0.0,  # discount_rate
    bender_cuts,
    1,  # years_per_stage
    1,  # n_cut_scenarios
)
simulations1, objectives1 = Stochastic.stochastic_planning(
    grid,
    scenarios,
    params,
    iteration_limit,
    n_simulations,
    SDDP.Expectation(),
)
simulations2, objectives2 = Stochastic.stochastic_planning(
    grid,
    scenarios,
    params,
    iteration_limit,
    n_simulations,
    SDDP.WorstCase(),
)
simulations3, objectives3 = Stochastic.stochastic_planning(
    grid,
    scenarios,
    params,
    iteration_limit,
    n_simulations,
    Wasserstein.risk_measure(1 / 20),
)

histogram(
    [objectives1, objectives2, objectives3],
    normalize = true,
    nbins = 100,
    xlabel = "Objective Value",
    ylabel = "Frequency",
    title = "Objective Distribution Across Scenarios",
    label = ["Without risk measure" "With risk measure WorstCase" "DRO Wasserstein"],
    opacity = 0.2,
    legend = :topright,
    fillalpha = 0.5,
    linecolor = [:blue :red :green],
    fillcolor = [:blue :red :green],
)
isdir(".cache") || mkpath(".cache")
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
