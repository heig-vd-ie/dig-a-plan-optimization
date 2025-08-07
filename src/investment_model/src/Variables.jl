
module Variables
using SDDP, JuMP

using ..Types

export define_state_variables!, define_decision_variables!

function define_state_variables!(m::Model, params::Types.PlanningParams, grid::Types.Grid)
    @variable(m, b_rem >= 0, SDDP.State, initial_value = params.initial_budget)  # Budget remaining
    @variable(m, cap[edge in grid.edges] >= 0, SDDP.State, initial_value = grid.initial_cap[edge])  # Capacity state variable
    @variable(m, δ_com[edge in grid.edges] >= 0, SDDP.State, initial_value = 0.0)  # committed expansion
    @variable(m, total_unmet_load[node in grid.nodes] >= 0, SDDP.State, initial_value = 0.0)  # unmet load state variable
    @variable(m, total_unmet_pv[node in grid.nodes] >= 0, SDDP.State, initial_value = 0.0)  # unmet PV state variable
    @variable(m, actual_load[node in grid.nodes] >= 0, SDDP.State, initial_value = grid.load[node])  # actual load state variable
    @variable(m, actual_pv[node in grid.nodes] >= 0, SDDP.State, initial_value = grid.pv[node])  # actual PV state variable
    return (
        b_rem = b_rem,
        cap = cap,
        δ_com = δ_com,
        total_unmet_load = total_unmet_load,
        total_unmet_pv = total_unmet_pv,
        actual_load = actual_load,
        actual_pv = actual_pv,
    )
end

function define_decision_variables!(m::Model, grid::Types.Grid)
    @variable(m, δ_cap[edge in grid.edges] >= 0)  # expansion decision
    @variable(m, investment_cost >= 0) # investment cost variable
    @variable(m, unmet_load[node in grid.nodes])  # unmet load variable
    @variable(m, unmet_pv[node in grid.nodes])  # unmet PV variable
    @variable(m, flow[edge in grid.edges])  # flow variable
    @variable(m, external_flow) # external flow variable
    @variable(m, obj) # objective value variable
    # Random variables (fixed by scenario)
    @variable(m, δ_load[node in grid.nodes])
    @variable(m, δ_pv[node in grid.nodes])
    @variable(m, δ_b)
    return (
        δ_cap = δ_cap,
        investment_cost = investment_cost,
        unmet_load = unmet_load,
        unmet_pv = unmet_pv,
        flow = flow,
        external_flow = external_flow,
        obj = obj,
        δ_load = δ_load,
        δ_pv = δ_pv,
        δ_b = δ_b,
    )
end

end
