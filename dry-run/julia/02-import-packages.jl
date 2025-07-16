# 02-import-packages.jl
using Pkg
Pkg.activate(dirname(@__DIR__))     # Activates the environment in the parent folder
Pkg.instantiate()                   # Installs all packages if not already installed

using JuMP, SDDP, Plots, Revise

println("Everything is ready!")