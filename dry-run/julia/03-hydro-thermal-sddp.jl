# 03-hydro-thermal-sddp.jl

using Pkg
Pkg.activate(dirname(@__DIR__))     # Activates the environment in the main folder
Pkg.instantiate()          # Installs all packages if not already installed

using SDDP, HiGHS

graph = SDDP.LinearGraph(3)

function subproblem_builder(subproblem::Model, node::Int)
    # State variables
    @variable(subproblem, 0 <= volume <= 200, SDDP.State, initial_value = 200)
    # Control variables
    @variables(subproblem, begin
        thermal_generation >= 0
        hydro_generation >= 0
        hydro_spill >= 0
    end)
    # Random variables
    @variable(subproblem, inflow)
    Ω = [0.0, 50.0, 100.0]
    P = [1 / 3, 1 / 3, 1 / 3]
    SDDP.parameterize(subproblem, Ω, P) do ω
        return JuMP.fix(inflow, ω)
    end
    # Transition function and constraints
    @constraints(
        subproblem,
        begin
            volume.out == volume.in - hydro_generation - hydro_spill + inflow
            demand_constraint, hydro_generation + thermal_generation == 150
        end
    )
    # Stage-objective
    fuel_cost = [50, 100, 150]
    @stageobjective(subproblem, fuel_cost[node] * thermal_generation)
    return subproblem
end

model = SDDP.LinearPolicyGraph(;
    stages = 3,
    sense = :Min,
    lower_bound = 0.0,
    optimizer = HiGHS.Optimizer,
) do subporblem, node
    subproblem_builder(subporblem, node)
end

SDDP.train(model; iteration_limit = 10)

rule = SDDP.DecisionRule(model; node = 1)

solution = SDDP.evaluate(
    rule;
    incoming_state = Dict(:volume => 150.0),
    noise = 50.0,
    controls_to_record = [:hydro_generation, :thermal_generation],
)

simulations = SDDP.simulate(
    # The trained model to simulate.
    model,
    # The number of replications.
    100,
    # A list of names to record the values of.
    [:volume, :thermal_generation, :hydro_generation, :hydro_spill],
)

replication = 1
stage = 2
simulations[replication][stage]

outgoing_volume = map(simulations[1]) do node
    return node[:volume].out
end

thermal_generation = map(simulations[1]) do node
    return node[:thermal_generation]
end

objectives = map(simulations) do simulation
    return sum(stage[:stage_objective] for stage in simulation)
end

μ, ci = SDDP.confidence_interval(objectives)
println("Confidence interval: ", μ, " ± ", ci)

println("Lower bound: ", SDDP.calculate_bound(model))

simulations = SDDP.simulate(
    model,
    1;  ## Perform a single simulation
    custom_recorders = Dict{Symbol,Function}(
        :price => (sp::JuMP.Model) -> JuMP.dual(sp[:demand_constraint]),
    ),
)

prices = map(simulations[1]) do node
    return node[:price]
end

V = SDDP.ValueFunction(model; node = 1)

cost, price = SDDP.evaluate(V, Dict("volume" => 10))
