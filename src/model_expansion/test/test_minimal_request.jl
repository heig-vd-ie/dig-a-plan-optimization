using HTTP
using JSON3
using Test

# Import server utilities
include("../src/ServerUtils.jl")
using .ServerUtils

# Server configuration from environment variables or command line arguments
server_config = get_server_config()
SERVER_BASE_URL = server_config.base_url

# Minimal request example - all parameters will use defaults
minimal_request = Dict()

# Simple request with just custom iteration limit
simple_request = Dict("iteration_limit" => 50, "n_simulations" => 100)

# Custom grid configuration
custom_request =
    JSON3.read(read(joinpath(@__DIR__, "../../../examples/payloads_jl/default.json"), String))
custom_request = Dict(custom_request)  # Convert to mutable Dict
custom_request[:additional_params] = Dict(
    "iteration_limit" => 50,
    "n_simulations" => 100,
    "risk_measure_type" => "Wasserstein",
    "risk_measure_param" => 0.1,
    "seed" => 1234,
)

function test_api_request(request_data, test_name)
    @testset "$test_name" begin
        response = HTTP.patch(
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


function test_generate_scenarios_request()
    @testset "Test generate scenarios" begin
        custom_request = JSON3.read(
            read(
                joinpath(@__DIR__, "../../../examples/payloads_jl/scenarios_request.json"),
                String,
            ),
        )
        custom_request = Dict(custom_request)
        response = HTTP.patch(
            "$SERVER_BASE_URL/generate-scenarios",
            ["Content-Type" => "application/json"],
            JSON3.write(custom_request),
        )

        @test response.status == 200

        if response.status == 200
            result = JSON3.read(String(response.body))
            @test haskey(result, "Ω")
            @test haskey(result, "P")
            @test length(result["Ω"]) > 0
            @test length(result["P"]) > 0
        end
    end
end

@testset "API Tests" begin
    # Run tests (make sure server is running first)
    test_api_request(minimal_request, "Minimal Request")
    test_api_request(simple_request, "Simple Request")
    test_api_request(custom_request, "Custom Request")
    test_generate_scenarios_request()
end
