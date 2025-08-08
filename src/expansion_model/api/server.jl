module API

using HTTP
using JSON3
using SDDP
using Random
using Dates
using ExpansionModel

function log_request(req::HTTP.Request, status_code::Int, processing_time::Float64)
    timestamp = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
    println(
        "[$timestamp] $(req.method) $(req.target) - Status: $status_code - Time: $(round(processing_time, digits=3))ms",
    )
end

function handle_stochastic_planning(req::HTTP.Request)
    println(
        "[$(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))] Processing stochastic planning request...",
    )

    # Parse the JSON body
    body = String(req.body)

    # Handle empty body case
    if isempty(body)
        println(
            "[$(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))] Empty request body, using all defaults",
        )
        input_data = Dict()
    else
        input_data = JSON3.read(body)
    end

    # Extract input parameters with default values
    # Read defaults from file if present
    default_config = JSON3.read(read(joinpath(@__DIR__, "../data/default.json"), String))
    # Grid structure with defaults
    grid_data = get(input_data, "grid", default_config["grid"])
    nodes_data = get(grid_data, "nodes", default_config["grid"]["nodes"])
    edges_data = get(grid_data, "edges", default_config["grid"]["edges"])
    cuts_data = get(grid_data, "cuts", default_config["grid"]["cuts"])
    external_grid_id = get(grid_data, "external_grid", default_config["grid"]["external_grid"])
    initial_cap = get(grid_data, "initial_cap", default_config["grid"]["initial_cap"])
    load = get(grid_data, "load", default_config["grid"]["load"])
    pv = get(grid_data, "pv", default_config["grid"]["pv"])

    # Scenarios with defaults
    scenarios_data = get(input_data, "scenarios", default_config["scenarios"])
    n_scenarios = get(scenarios_data, "n_scenarios", default_config["scenarios"]["n_scenarios"])
    n_stages = get(scenarios_data, "n_stages", default_config["scenarios"]["n_stages"])
    total_load_per_node = Float64(
        get(
            scenarios_data,
            "total_load_per_node",
            default_config["scenarios"]["total_load_per_node"],
        ),
    )
    total_pv_per_node = Float64(
        get(
            scenarios_data,
            "total_pv_per_node",
            default_config["scenarios"]["total_pv_per_node"],
        ),
    )
    total_budget = Float64(
        get(scenarios_data, "total_budget", default_config["scenarios"]["total_budget"]),
    )
    seed_number = get(scenarios_data, "seed", default_config["scenarios"]["seed"])

    # Planning parameters with defaults
    params_data = get(input_data, "params", Dict())
    initial_budget =
        Float64(get(params_data, "initial_budget", default_config["params"]["initial_budget"]))
    discount_rate =
        Float64(get(params_data, "discount_rate", default_config["params"]["discount_rate"]))
    investment_cost_range = get(
        params_data,
        "investment_cost_range",
        default_config["params"]["investment_cost_range"],
    )
    penalty_cost_load = Float64(
        get(params_data, "penalty_cost_load", default_config["params"]["penalty_cost_load"]),
    )
    penalty_cost_pv = Float64(
        get(params_data, "penalty_cost_pv", default_config["params"]["penalty_cost_pv"]),
    )

    # Algorithm parameters with defaults
    iteration_limit = get(input_data, "iteration_limit", default_config["iteration_limit"])
    n_simulations = get(input_data, "n_simulations", default_config["n_simulations"])
    risk_measure_type = get(input_data, "risk_measure", default_config["risk_measure"])
    risk_measure_param =
        Float64(get(input_data, "risk_measure_param", default_config["risk_measure_param"]))

    println("[$(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))] Running SDDP optimization...")

    # Construct data structures from parsed parameters
    # Create nodes, edges, and cuts
    nodes = [ExpansionModel.Types.Node(node_data["id"]) for node_data in nodes_data]
    edges = [
        ExpansionModel.Types.Edge(edge_data["id"], edge_data["from"], edge_data["to"]) for
        edge_data in edges_data
    ]
    cuts = [ExpansionModel.Types.Cut(cut_data["id"]) for cut_data in cuts_data]
    external_grid = ExpansionModel.Types.Node(external_grid_id)

    # Convert initial_cap, load, pv to use proper types (find edge/node by id)
    # Initialize with defaults for all edges/nodes, then override with provided values
    initial_cap_dict = Dict(edge => 1.0 for edge in edges)  # Default capacity
    for (k, v) in initial_cap
        edge_id = parse(Int, string(k))
        edge_idx = findfirst(e -> e.id == edge_id, edges)
        if edge_idx !== nothing
            initial_cap_dict[edges[edge_idx]] = v
        end
    end

    load_dict = Dict(node => 1.0 for node in nodes)  # Default load
    for (k, v) in load
        node_id = parse(Int, string(k))
        node_idx = findfirst(n -> n.id == node_id, nodes)
        if node_idx !== nothing
            load_dict[nodes[node_idx]] = v
        end
    end

    pv_dict = Dict(node => 0.1 for node in nodes)  # Default PV
    for (k, v) in pv
        node_id = parse(Int, string(k))
        node_idx = findfirst(n -> n.id == node_id, nodes)
        if node_idx !== nothing
            pv_dict[nodes[node_idx]] = v
        end
    end

    # Create Grid
    grid = ExpansionModel.Types.Grid(
        nodes,
        edges,
        cuts,
        external_grid,
        initial_cap_dict,
        load_dict,
        pv_dict,
    )

    # Generate scenarios
    Ω = ExpansionModel.ScenariosGeneration.generate_scenarios(
        n_scenarios,
        n_stages,
        nodes;
        total_load_per_node = total_load_per_node,
        total_pv_per_node = total_pv_per_node,
        total_budget = total_budget,
        seed_number = seed_number,
    )
    P = fill(1.0 / n_scenarios, n_scenarios)
    scenarios = ExpansionModel.Types.Scenarios(Ω, P)

    # Generate costs and Bender cuts
    investment_costs =
        Dict(e => rand(investment_cost_range[1]:investment_cost_range[2]) for e in edges)
    penalty_costs_load = Dict(n => penalty_cost_load for n in nodes)
    penalty_costs_pv = Dict(n => penalty_cost_pv for n in nodes)

    # Generate simple Bender cuts (you may need to adjust this based on your needs)
    bender_cuts = Dict(
        cut => ExpansionModel.Types.BenderCut(
            0.0,  # θ
            Dict(edge => edge.id == cut.id ? -1.0 : 0.0 for edge in edges),  # λ_cap
            Dict(node => 0.0 for node in nodes),  # λ_load
            Dict(node => 0.0 for node in nodes),  # λ_pv
            Dict(edge => 0.0 for edge in edges),  # cap0
            Dict(node => 0.0 for node in nodes),  # load0
            Dict(node => 0.0 for node in nodes),   # pv0
        ) for cut in cuts
    )

    # Create PlanningParams
    params = ExpansionModel.Types.PlanningParams(
        n_stages,
        initial_budget,
        investment_costs,
        penalty_costs_load,
        penalty_costs_pv,
        discount_rate,
        bender_cuts,
    )

    # Create risk measure
    risk_measure = if risk_measure_type == "expectation"
        SDDP.Expectation()
    elseif risk_measure_type == "entropic"
        SDDP.Entropic(risk_measure_param)
    else
        SDDP.Expectation()  # default
    end

    # Call your stochastic_planning function
    simulations, objectives = ExpansionModel.Stochastic.stochastic_planning(
        grid,
        scenarios,
        params,
        iteration_limit,
        n_simulations,
        risk_measure,
    )

    println("[$(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))] SDDP optimization completed")

    # Return the response as JSON
    response = JSON3.write(Dict("simulations" => simulations, "objectives" => objectives))

    return HTTP.Response(200, response)
end

function handle_request(req::HTTP.Request)
    start_time = time()

    try
        if req.method == "POST" && req.target == "/stochastic_planning"
            log_request(req, 500, (time() - start_time) * 1000)
            return handle_stochastic_planning(req)
            log_request(req, 200, (time() - start_time) * 1000)
        elseif req.method == "GET" && req.target == "/health"
            log_request(req, 200, (time() - start_time) * 1000)
            return HTTP.Response(200, "API is running")
        else
            log_request(req, 404, (time() - start_time) * 1000)
            return HTTP.Response(404, "Not Found")
        end
    catch e
        log_request(req, 500, (time() - start_time) * 1000)
        println("[$(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))] ERROR: $e")
        log_request(req, 500, (time() - start_time) * 1000)
        return HTTP.Response(500, "Internal Server Error")
    end
end

# Parse command line arguments or environment variables
SERVER_HOST = "0.0.0.0"

SERVER_PORT = if length(ARGS) >= 1
    parse(Int, ARGS[1])
elseif haskey(ENV, "SERVER_PORT")
    parse(Int, ENV["SERVER_PORT"])
else
    8080
end

println(
    "[$(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))] Starting server on $SERVER_HOST:$SERVER_PORT...",
)
HTTP.serve(handle_request, SERVER_HOST, SERVER_PORT)

end
