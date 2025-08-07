module Constraints
using SDDP, JuMP

using ..Types, ..Variables

export define_constraints!

function define_constraints!(m::Model, grid::Types.Grid, vars, params::Types.PlanningParams)
    return nothing
end
end