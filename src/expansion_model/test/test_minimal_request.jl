using HTTP
using JSON3
using Test

# Expected JSON structure:
# {
#   "grid": {
#     "nodes": [{"id": 1}, {"id": 2}],
#     "edges": [{"id": 1, "from": 1, "to": 2}],
#     "cuts": [{"id": 1}],
#     "external_grid": 1,
#     "initial_cap": {"1": 1.0},
#     "load": {"1": 1.0, "2": 1.0},
#     "pv": {"1": 0.1, "2": 0.1}
#   },
#   "scenarios": {
#     "n_scenarios": 10,
#     "n_stages": 5,
#     "total_load_per_node": 2.0,
#     "total_pv_per_node": 1.0,
#     "total_budget": 1000.0,
#     "seed": 1234
#   },
#   "params": {
#     "initial_budget": 50.0,
#     "discount_rate": 0.0,
#     "investment_cost_range": [90.0, 100.0],
#     "penalty_cost_load": 6000.0,
#     "penalty_cost_pv": 6000.0
#   },
#   "iteration_limit": 100,
#   "n_simulations": 1000,
#   "risk_measure": "expectation",
#   "risk_measure_param": 0.1
# }

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
    @testset "$test_name" begin
        response = HTTP.post(
            "http://localhost:8080/stochastic_planning",
            ["Content-Type" => "application/json"],
            JSON3.write(request_data),
        )

        @test response.status == 200

        if response.status == 200
            result = JSON3.read(String(response.body))
            @test haskey(result, "objectives")
            @test haskey(result, "simulations")
            @test length(result["objectives"]) > 0
            @test length(result["simulations"]) > 0
            @test length(result["objectives"]) == length(result["simulations"])
        end
    end
end

@testset "API Tests" begin
    # Run tests (make sure server is running first)
    test_api_request(minimal_request, "Minimal Request")
    test_api_request(simple_request, "Simple Request")
    test_api_request(custom_request, "Custom Request")
end
