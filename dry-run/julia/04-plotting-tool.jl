using Pkg
Pkg.activate(dirname(@__DIR__))     # Activates the environment in the main folder
Pkg.instantiate()          # Installs all packages if not already installed

using SDDP, HiGHS

Ω = [
    (inflow = 0.0, fuel_multiplier = 1.5),
    (inflow = 50.0, fuel_multiplier = 1.0),
    (inflow = 100.0, fuel_multiplier = 0.75),
]

model = SDDP.MarkovianPolicyGraph(;
    transition_matrices = Array{Float64,2}[
        [1.0]',
        [0.75 0.25],
        [0.75 0.25; 0.25 0.75],
    ],
    sense = :Min,
    lower_bound = 0.0,
    optimizer = HiGHS.Optimizer,
) do subproblem, node
    t, markov_state = node
    @variable(subproblem, 0 <= volume <= 200, SDDP.State, initial_value = 200)
    @variables(subproblem, begin
        thermal_generation >= 0
        hydro_generation >= 0
        hydro_spill >= 0
        inflow
    end)
    @constraints(
        subproblem,
        begin
            volume.out == volume.in + inflow - hydro_generation - hydro_spill
            thermal_generation + hydro_generation == 150.0
        end
    )
    probability =
        markov_state == 1 ? [1 / 6, 1 / 3, 1 / 2] : [1 / 2, 1 / 3, 1 / 6]
    fuel_cost = [50.0, 100.0, 150.0]
    SDDP.parameterize(subproblem, Ω, probability) do ω
        JuMP.fix(inflow, ω.inflow)
        @stageobjective(
            subproblem,
            ω.fuel_multiplier * fuel_cost[t] * thermal_generation
        )
    end
end

SDDP.train(model; iteration_limit = 20, run_numerical_stability_report = false)

simulations = SDDP.simulate(
    model,
    100,
    [:volume, :thermal_generation, :hydro_generation, :hydro_spill],
)

println("Completed $(length(simulations)) simulations.")

plt = SDDP.SpaghettiPlot(simulations)

SDDP.add_spaghetti(plt; title = "Reservoir volume") do data
    return data[:volume].out
end


SDDP.add_spaghetti(plt; title = "Fuel cost", ymin = 0, ymax = 250) do data
    if data[:thermal_generation] > 0
        return data[:stage_objective] / data[:thermal_generation]
    else  # No thermal generation, so return 0.0.
        return 0.0
    end
end

SDDP.plot(plt, "dry-run/04-spaghetti_plot.html", open = false)

import Plots
Plots.plot(
    SDDP.publication_plot(simulations; title = "Outgoing volume") do data
        return data[:volume].out
    end,
    SDDP.publication_plot(simulations; title = "Thermal generation") do data
        return data[:thermal_generation]
    end;
    xlabel = "Stage",
    ylims = (0, 200),
    layout = (1, 2),
)

Plots.savefig("dry-run/04-spaghetti_plot.pdf")

