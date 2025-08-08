using HTTP
using JSON3

# Minimal request example - all parameters will use defaults
minimal_request = Dict()

# Simple request with just custom iteration limit
simple_request = Dict("iteration_limit" => 50, "n_simulations" => 100)

# Custom grid configuration
custom_request = Dict(
    "grid" => Dict(
        "nodes" => [Dict("id" => 1), Dict("id" => 2), Dict("id" => 3)],
        "edges" => [
            Dict("id" => 1, "from" => 1, "to" => 2),
            Dict("id" => 2, "from" => 2, "to" => 3),
        ],
        "cuts" => [Dict("id" => 1), Dict("id" => 2)],
        "external_grid" => 1,
        "initial_cap" => Dict("1" => 2.0, "2" => 1.5),
        "load" => Dict("1" => 1.5, "2" => 1.2, "3" => 1.0),
        "pv" => Dict("1" => 0.2, "2" => 0.15, "3" => 0.1),
    ),
    "scenarios" => Dict(
        "n_scenarios" => 20,
        "n_stages" => 10,
        "total_load_per_node" => 3.0,
        "total_pv_per_node" => 1.5,
        "total_budget" => 2000.0,
        "seed" => 5678,
    ),
    "params" => Dict(
        "initial_budget" => 100.0,
        "discount_rate" => 0.05,
        "investment_cost_range" => [80.0, 120.0],
        "penalty_cost_load" => 8000.0,
        "penalty_cost_pv" => 8000.0,
    ),
    "iteration_limit" => 200,
    "n_simulations" => 500,
    "risk_measure" => "entropic",
    "risk_measure_param" => 0.2,
)

function test_api_request(request_data, test_name)
    println("Testing $test_name...")
    try
        response = HTTP.post(
            "http://localhost:8080/stochastic_planning",
            ["Content-Type" => "application/json"],
            JSON3.write(request_data),
        )

        if response.status == 200
            result = JSON3.read(String(response.body))
            println("✅ $test_name successful!")
            println("   Objectives length: $(length(result["objectives"]))")
            println("   Simulations length: $(length(result["simulations"]))")
        else
            println("❌ $test_name failed with status: $(response.status)")
        end
    catch e
        println("❌ $test_name failed with error: $e")
    end
    println()
end

# Uncomment to run tests (make sure server is running first)
test_api_request(minimal_request, "Minimal Request")
test_api_request(simple_request, "Simple Request")
test_api_request(custom_request, "Custom Request")
