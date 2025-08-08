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
    # Grid structure with defaults
    grid_data = get(input_data, "grid", Dict())
    nodes_data = get(grid_data, "nodes", [Dict("id" => 1), Dict("id" => 2)])
    edges_data = get(grid_data, "edges", [Dict("id" => 1, "from" => 1, "to" => 2)])
    cuts_data = get(grid_data, "cuts", [Dict("id" => 1)])
    external_grid_id = get(grid_data, "external_grid", 1)
    initial_cap = get(grid_data, "initial_cap", Dict("1" => 1.0))
    load = get(grid_data, "load", Dict("1" => 1.0, "2" => 1.0))
    pv = get(grid_data, "pv", Dict("1" => 0.1, "2" => 0.1))

    # Scenarios with defaults
    scenarios_data = get(input_data, "scenarios", Dict())
    n_scenarios = get(scenarios_data, "n_scenarios", 10)
    n_stages = get(scenarios_data, "n_stages", 5)
    total_load_per_node = Float64(get(scenarios_data, "total_load_per_node", 2.0))
    total_pv_per_node = Float64(get(scenarios_data, "total_pv_per_node", 1.0))
    total_budget = Float64(get(scenarios_data, "total_budget", 1000.0))
    seed_number = get(scenarios_data, "seed", 1234)

    # Planning parameters with defaults
    params_data = get(input_data, "params", Dict())
    initial_budget = Float64(get(params_data, "initial_budget", 50.0))
    discount_rate = Float64(get(params_data, "discount_rate", 0.0))
    investment_cost_range = get(params_data, "investment_cost_range", [90.0, 100.0])
    penalty_cost_load = Float64(get(params_data, "penalty_cost_load", 6000.0))
    penalty_cost_pv = Float64(get(params_data, "penalty_cost_pv", 6000.0))

    # Algorithm parameters with defaults
    iteration_limit = get(input_data, "iteration_limit", 100)
    n_simulations = get(input_data, "n_simulations", 1000)
    risk_measure_type = get(input_data, "risk_measure", "expectation")
    risk_measure_param = Float64(get(input_data, "risk_measure_param", 0.1))

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
port = if length(ARGS) >= 1
    parse(Int, ARGS[1])
else
    8080
end

host = if length(ARGS) >= 2
    ARGS[2]
else
    "0.0.0.0"
end

println("[$(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))] Starting server on $host:$port...")
HTTP.serve(handle_request, host, port)

end
