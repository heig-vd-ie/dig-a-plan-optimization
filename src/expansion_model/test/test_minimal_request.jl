using HTTP
using JSON3
using Test

# Server configuration from environment variables or command line arguments
SERVER_HOST = "localhost"

SERVER_PORT = if length(ARGS) >= 1
    parse(Int, ARGS[1])
elseif haskey(ENV, "SERVER_PORT")
    parse(Int, ENV["SERVER_PORT"])
else
    8080
end

SERVER_BASE_URL = "http://$SERVER_HOST:$SERVER_PORT"

println("Testing API server at: $SERVER_BASE_URL")

# Minimal request example - all parameters will use defaults
minimal_request = Dict()

# Simple request with just custom iteration limit
simple_request = Dict("iteration_limit" => 50, "n_simulations" => 100)

# Custom grid configuration
custom_request = JSON3.read(read(joinpath(@__DIR__, "../data/default.json"), String))
custom_request = Dict(custom_request)  # Convert to mutable Dict
custom_request[:additional_params] = Dict(
    "iteration_limit" => 50,
    "n_simulations" => 100,
    "risk_measure_type" => "wasserstein",
    "risk_measure_param" => 0.1,
)

function test_api_request(request_data, test_name)
    @testset "$test_name" begin
        response = HTTP.post(
            "$SERVER_BASE_URL/stochastic_planning",
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
