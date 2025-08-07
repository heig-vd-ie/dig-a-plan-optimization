module InvestmentModel

using SDDP

include("Types.jl")
include("Variables.jl")
include("Stochastic.jl")

export Types, Stochastic, Variables

end