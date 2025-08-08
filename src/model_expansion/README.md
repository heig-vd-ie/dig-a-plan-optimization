# Stochastic Distribution Grid Planning

This project provides a stochastic programming framework for expansion, reinforcement, and replacement planning of distribution grids using SDDP.jl and JuMP.jl.

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

   - [docs/Julia/01-install-julia.md]()
   - [docs/Julia/02-create-julia-projects]()
   - [docs/Julia/03-hydro-thermal-sddp.md]()
