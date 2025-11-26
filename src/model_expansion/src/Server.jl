module API

using HTTP
using JSON3
using SDDP
using Random
using Dates
using ExpansionModel
using Plots
using ProgressMeter

include("ServerUtils.jl")
using .ServerUtils

function log_request(req::HTTP.Request, status_code::Int, processing_time::Float64)
    timestamp = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
    println(
        "[$timestamp] $(req.method) $(req.target) - Status: $status_code - Time: $(round(processing_time, digits=3))ms",
    )
end

function log_datetime()
    return Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
end

function handle_generate_scenarios(req::HTTP.Request)
    println("[$(log_datetime())] Processing generate scenario request...")

    body = String(req.body)
    # Handle empty body case
    if isempty(body)
        println("[$(log_datetime())] Empty request body, using all defaults")
        body = Dict()
    else
        body = JSON3.read(body)
    end

    n_scenarios = get(body, "n_scenarios", 1)
    n_stages = get(body, "n_stages", 1)
    _nodes = get(body, "nodes", 1)
    _load_potential = get(body, "load_potential", 1)
    _pv_potential = get(body, "pv_potential", 1)
    nodes = Vector{ExpansionModel.Types.Node}()
    load_potential, pv_potential =
        Dict{ExpansionModel.Types.Node, Float64}(), Dict{ExpansionModel.Types.Node, Float64}()
    @showprogress 1 "Processing nodes..." for node_data in _nodes
        node = ExpansionModel.Types.Node(node_data["id"])
        node_id_str = string(node_data["id"])
        push!(nodes, node)
        load_potential[node] = Float64(_load_potential[node_id_str])
        pv_potential[node] = Float64(_pv_potential[node_id_str])
    end

    min_load = Float64(get(body, "min_load", 1.0))
    min_pv = Float64(get(body, "min_pv", 5.0))
    yearly_budget = Float64(get(body, "yearly_budget", 1))
    N_years_per_stage = get(body, "N_years_per_stage", 1)
    seed_number = get(body, "seed_number", 1234)

    scenarios = ExpansionModel.ScenariosGeneration.generate_scenarios(
        n_scenarios,
        n_stages,
        nodes,
        load_potential,
        pv_potential,
        min_load,
        min_pv,
        yearly_budget,
        N_years_per_stage,
        seed_number,
    )
    probabilities = fill(1.0 / n_scenarios, n_scenarios)
    response = JSON3.write(Dict("Ω" => scenarios, "P" => probabilities))
    response = replace(response, r"ExpansionModel\.Types\.Node\((\d+)\)" => s"\1")
    response = replace(response, r"Node\((\d+)\)" => s"\1")
    return HTTP.Response(200, response)
end

function handle_stochastic_planning(req::HTTP.Request)
    println("[$(log_datetime())] Processing stochastic planning request...")

    # Parse the JSON body
    body = String(req.body)

    # Handle empty body case
    if isempty(body)
        println("[$(log_datetime())] Empty request body, using all defaults")
        body = Dict()
    else
        body = JSON3.read(body)
    end

    # Extract input parameters with default values
    # Read defaults from file if present
    default =
        JSON3.read(read(joinpath(@__DIR__, "..", "..", "..", "examples", "default.json"), String))
    # Grid structure with defaults
    grid_data = get(body, "grid", default["grid"])
    scenarios_folder = get(body, "scenarios", default["scenarios"])
    scenarios_data =
        JSON3.read(read(joinpath(@__DIR__, "..", "..", "..", scenarios_folder), String))
    out_of_sample_scenarios_folder =
        get(body, "out_of_sample_scenarios", default["out_of_sample_scenarios"])
    bender_cuts_folder = get(body, "bender_cuts", default["bender_cuts"])
    planning_params = get(body, "planning_params", default["planning_params"])

    # Algorithm parameters with defaults
    additional_params = get(body, "additional_params", default["additional_params"])

    println("[$(log_datetime())] Running SDDP parameters initialization...")

    # Construct data structures from parsed parameters
    # Create nodes, edges, and cuts

    nodes, edges, cuts = [], [], []
    initial_cap_dict, load_dict, pv_dict = Dict(), Dict(), Dict()

    @showprogress 1 "Processing nodes..." for node_data in grid_data["nodes"]
        node_id_str = string(node_data["id"])
        node = ExpansionModel.Types.Node(node_data["id"])
        push!(nodes, node)
        load_dict[node] = grid_data["load"][node_id_str]
        pv_dict[node] = grid_data["pv"][node_id_str]
    end

    @showprogress 1 "Processing edges..." for edge_data in grid_data["edges"]
        edge_id_str = string(edge_data["id"])
        edge =
            ExpansionModel.Types.Edge(edge_data["id"], edge_data["source"], edge_data["target"])
        push!(edges, edge)
        initial_cap_dict[edge] = grid_data["initial_cap"][edge_id_str]
    end

    @showprogress 1 "Processing cuts..." for cut_data in grid_data["cuts"]
        push!(cuts, ExpansionModel.Types.Cut(cut_data["id"]))
    end
    external_grid = ExpansionModel.Types.Node(grid_data["external_grid"])

    println("[$(log_datetime())] Running SDDP parameters initialization5...")

    Ω = [
        [
            ExpansionModel.Types.Scenario(
                Dict(node => scenario1["δ_load"][string(node.id)] for node in nodes),
                Dict(node => scenario1["δ_pv"][string(node.id)] for node in nodes),
                scenario1["δ_b"],
            ) for scenario1 in scenario2
        ] for scenario2 in scenarios_data["Ω"]
    ]
    println("[$(log_datetime())] Running SDDP parameters initialization6...")

    P = [scenarios_data["P"][i] for i in 1:length(scenarios_data["P"])]
    println("[$(log_datetime())] Running SDDP parameters initialization7...")

    scenarios = ExpansionModel.Types.Scenarios(Ω, P)
    println("[$(log_datetime())] Running SDDP parameters initialization8...")

    out_of_sample_scenarios_data = JSON3.read(
        read(joinpath(@__DIR__, "..", "..", "..", out_of_sample_scenarios_folder), String),
    )
    println("[$(log_datetime())] Running SDDP parameters initialization9...")

    Ω_out = [
        [
            ExpansionModel.Types.Scenario(
                Dict(node => scenario1["δ_load"][string(node.id)] for node in nodes),
                Dict(node => scenario1["δ_pv"][string(node.id)] for node in nodes),
                scenario1["δ_b"],
            ) for scenario1 in scenario2
        ] for scenario2 in out_of_sample_scenarios_data["Ω"]
    ]
    println("[$(log_datetime())] Running SDDP parameters initialization10...")

    P_out = [
        out_of_sample_scenarios_data["P"][i] for
        i in 1:length(out_of_sample_scenarios_data["P"])
    ]
    println("[$(log_datetime())] Running SDDP parameters initialization11...")
    out_of_sample_scenarios = ExpansionModel.Types.Scenarios(Ω_out, P_out)
    println("[$(log_datetime())] Running SDDP parameters initialization12...")

    n_stages = planning_params["n_stages"]
    years_per_stage = planning_params["years_per_stage"]
    n_cut_scenarios = planning_params["n_cut_scenarios"]
    initial_budget = planning_params["initial_budget"]
    γ_cuts = planning_params["γ_cuts"]
    println("[$(log_datetime())] Running SDDP parameters initialization13...")
    investment_costs =
        Dict(edge => grid_data["investment_costs"][string(edge.id)] for edge in edges)
    penalty_costs_load =
        Dict(node => grid_data["penalty_costs_load"][string(node.id)] for node in nodes)
    penalty_costs_pv =
        Dict(node => grid_data["penalty_costs_pv"][string(node.id)] for node in nodes)
    penalty_costs_infeasibility = grid_data["penalty_costs_infeasibility"]

    println("[$(log_datetime())] Running SDDP parameters initialization14...")

    discount_rate = planning_params["discount_rate"]
    bender_cuts_data =
        JSON3.read(read(joinpath(@__DIR__, "..", "..", "..", bender_cuts_folder), String))

    println("[$(log_datetime())] Running SDDP parameters initialization15...")

    # Generate simple Bender cuts (you may need to adjust this based on your needs)
    bender_cuts = Dict(
        cut => ExpansionModel.Types.BenderCut(
            bender_cuts_data["cuts"][string(cut.id)]["θ"],
            Dict(
                edge => bender_cuts_data["cuts"][string(cut.id)]["λ_cap"][string(edge.id)]
                for edge in edges
            ),
            Dict(
                node => bender_cuts_data["cuts"][string(cut.id)]["λ_load"][string(node.id)]
                for node in nodes
            ),
            Dict(
                node => bender_cuts_data["cuts"][string(cut.id)]["λ_pv"][string(node.id)]
                for node in nodes
            ),
            Dict(
                edge => bender_cuts_data["cuts"][string(cut.id)]["cap0"][string(edge.id)]
                for edge in edges
            ),
            Dict(
                node => bender_cuts_data["cuts"][string(cut.id)]["load0"][string(node.id)]
                for node in nodes
            ),
            Dict(
                node => bender_cuts_data["cuts"][string(cut.id)]["pv0"][string(node.id)] for
                node in nodes
            ),
        ) for cut in cuts
    )
    println("[$(log_datetime())] Running SDDP parameters initialization16...")

    # Create PlanningParams
    params = ExpansionModel.Types.PlanningParams(
        n_stages,
        initial_budget,
        γ_cuts,
        investment_costs,
        penalty_costs_load,
        penalty_costs_pv,
        penalty_costs_infeasibility,
        discount_rate,
        bender_cuts,
        years_per_stage,
        n_cut_scenarios,
    )

    iteration_limit = additional_params["iteration_limit"]
    n_simulations = additional_params["n_simulations"]
    risk_measure_type = additional_params["risk_measure_type"]
    risk_measure_param = additional_params["risk_measure_param"]

    println("[$(log_datetime())] Running SDDP parameters initialization17...")

    # Create risk measure
    risk_measure = if risk_measure_type == "Expectation"
        SDDP.Expectation()
    elseif risk_measure_type == "Entropic"
        SDDP.Entropic(risk_measure_param)
    elseif risk_measure_type == "Wasserstein"
        ExpansionModel.Wasserstein.risk_measure(risk_measure_param)
    elseif risk_measure_type == "CVaR"
        SDDP.CVaR(risk_measure_param)
    elseif risk_measure_type == "WorstCase"
        SDDP.WorstCase()
    else
        SDDP.Expectation()  # default
    end

    println("[$(log_datetime())] Running SDDP optimization...")
    # Call your stochastic_planning function
    grid = ExpansionModel.Types.Grid(
        nodes,
        edges,
        cuts,
        external_grid,
        initial_cap_dict,
        load_dict,
        pv_dict,
    )

    simulations, objectives, out_of_sample_simulations, out_of_sample_objectives =
        ExpansionModel.Stochastic.stochastic_planning(
            grid,
            scenarios,
            out_of_sample_scenarios,
            params,
            iteration_limit,
            n_simulations,
            risk_measure,
            additional_params["seed"],
        )

    println("[$(log_datetime())] SDDP optimization completed")

    # Return the response as JSON
    response = JSON3.write(
        Dict(
            "simulations" => simulations,
            "objectives" => objectives,
            "out_of_sample_simulations" => out_of_sample_simulations,
            "out_of_sample_objectives" => out_of_sample_objectives,
        ),
    )

    return HTTP.Response(200, response)
end

function compare_plot(req::HTTP.Request)
    println("[$(log_datetime())] Processing compare plot request...")

    # Parse the original request body
    body = String(req.body)
    if isempty(body)
        body = Dict()
    else
        body = JSON3.read(body)
        body = Dict(body)  # Convert to mutable Dict
    end

    cases = [Dict(case) for case in body[:cases]]  # Convert each case to Dict

    objectives_list = []
    responses = []

    # Run stochastic planning for each case
    for (i, case_params) in enumerate(cases)
        println(
            "[$(log_datetime())] Running case $i with risk measure: $(case_params[:risk_measure_type])",
        )

        # Create modified body with the current case parameters
        modified_body = deepcopy(body)
        modified_body[:additional_params] = case_params

        # Create a new request with modified body
        modified_req =
            HTTP.Request("PATCH", "/stochastic_planning", [], JSON3.write(modified_body))

        # Run stochastic planning
        response = handle_stochastic_planning(modified_req)
        push!(responses, response)

        # Extract objectives from response
        response_data = JSON3.read(String(response.body))
        objectives = response_data["objectives"]
        push!(objectives_list, objectives)
    end

    # Create comparison histogram
    println("[$(log_datetime())] Creating comparison histogram...")
    histogram(
        objectives_list,
        normalize = true,
        nbins = 100,
        xlabel = "Objective Value",
        ylabel = "Frequency",
        title = "Objective Distribution Across Scenarios",
        label = reshape([case[:risk_measure_type] for case in cases], 1, :),
        alpha = 0.7,
        legend = :topright,
        linecolor = [:blue :red :green],
        fillcolor = [:blue :red :green],
    )

    # Save the plot
    isdir(".cache") || mkpath(".cache")
    savefig(get(body, :plot_saved, ".cache/objective_histogram.pdf"))

    println("[$(log_datetime())] Comparison plot completed and saved")

    return HTTP.Response(200, JSON3.write(Dict("message" => "Plot generated successfully")))
end

function handle_request(req::HTTP.Request)
    start_time = time()

    try
        if req.method == "PATCH" && req.target == "/stochastic_planning"
            response = handle_stochastic_planning(req)
            log_request(req, 200, (time() - start_time) * 1000)
            return response
        elseif req.method == "PATCH" && req.target == "/compare-plot"
            # Handle plot request
            response = compare_plot(req)
            log_request(req, 200, (time() - start_time) * 1000)
            return response
        elseif req.method == "PATCH" && req.target == "/generate-scenarios"
            response = handle_generate_scenarios(req)
            log_request(req, 200, (time() - start_time) * 1000)
            return response
        elseif req.method == "GET" && req.target == "/health"
            log_request(req, 200, (time() - start_time) * 1000)
            return HTTP.Response(200, "API is running")
        else
            log_request(req, 404, (time() - start_time) * 1000)
            return HTTP.Response(404, "Not Found")
        end
    catch e
        log_request(req, 500, (time() - start_time) * 1000)
        println("[$(log_datetime())] ERROR: $e")
        return HTTP.Response(500, "Internal Server Error")
    end
end

# Parse command line arguments or environment variables
server_config = get_server_config()

println("[$(log_datetime())] Starting server on $(server_config.host):$(server_config.port)...")
HTTP.serve(handle_request, server_config.host, server_config.port)

end
