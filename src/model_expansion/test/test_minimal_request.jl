using HTTP
using JSON3
using Test

# Import server utilities
include("../src/ServerUtils.jl")
using .ServerUtils

# Server configuration from environment variables or command line arguments
server_config = get_server_config("localhost", 8080, verbose = true)
SERVER_BASE_URL = server_config.base_url

# Minimal request example - all parameters will use defaults
minimal_request = Dict()

# Simple request with just custom iteration limit
simple_request = Dict("iteration_limit" => 50, "n_simulations" => 100)

# Custom grid configuration
custom_request = JSON3.read(read(joinpath(@__DIR__, "../../../data/default.json"), String))
custom_request = Dict(custom_request)  # Convert to mutable Dict
custom_request[:additional_params] = Dict(
    "iteration_limit" => 50,
    "n_simulations" => 100,
    "risk_measure_type" => "Wasserstein",
    "risk_measure_param" => 0.1,
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

function test_plot()
    @testset "Plot Tests" begin
        custom_request =
            JSON3.read(read(joinpath(@__DIR__, "../../../data/default.json"), String))
        custom_request = Dict(custom_request)  # Convert to mutable Dict
        custom_request[:cases] = [
            Dict(
                "iteration_limit" => 50,
                "n_simulations" => 100,
                "risk_measure_type" => "Expectation",
                "risk_measure_param" => 0.1,
                "seed" => 1234,
            ),
            Dict(
                "iteration_limit" => 50,
                "n_simulations" => 100,
                "risk_measure_type" => "Entropic",
                "risk_measure_param" => 0.1,
                "seed" => 1234,
            ),
            Dict(
                "iteration_limit" => 50,
                "n_simulations" => 100,
                "risk_measure_type" => "Wasserstein",
                "risk_measure_param" => 0.1,
                "seed" => 1234,
            ),
        ]
        custom_request[:plot_saved] = ".cache/objective_histogram.pdf"
        response = HTTP.patch(
            "$SERVER_BASE_URL/compare-plot",
            ["Content-Type" => "application/json"],
            JSON3.write(custom_request),
        )
        @test response.status == 200
    end
end

@testset "API Tests" begin
    # Run tests (make sure server is running first)
    test_api_request(minimal_request, "Minimal Request")
    test_api_request(simple_request, "Simple Request")
    test_api_request(custom_request, "Custom Request")
    test_plot()
end
