# Stochastic Distribution Grid Planning

This project provides a stochastic programming framework for expansion, reinforcement, and replacement planning of distribution grids using SDDP.jl and JuMP.jl.

## Project Structure

- `src/types.jl`: Defines core data structures (`Scenario`, `PlanningParams`).
- `src/utils.jl`: Utility functions for scenario/cost generation and plotting.
- `src/stochastic.jl`: Main SDDP model-building logic.
- `examples/stochastic_example.jl`: Example script demonstrating usage.
- `examples/entrypoint.jl`: Entrypoint script that sets up the environment and runs the example.

## Usage

1. Install dependencies:
   ```julia
   import Pkg; Pkg.instantiate()
   ```
2. Run the example via the entrypoint:
   ```julia
   julia examples/entrypoint.jl
   ```
   This ensures all modules in `src/` are available to the example and any additional scripts you include.

## Documentation
- See the `docs/` folder for installation and project guides.
