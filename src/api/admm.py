from data_exporter.kace_to_dap import kace4reconfiguration
from data_model.reconfiguration import ADMMInput, ReconfigurationOutput
from pipelines.reconfiguration.configs import ADMMConfig
from pipelines.reconfiguration import DigAPlanADMM
from data_exporter.dap_to_mock import save_dap_state


def run_admm(request: ADMMInput) -> ReconfigurationOutput:
    base_grid_data = kace4reconfiguration(
        grid=request.grid,
        load_profiles=request.load_profiles,
        st_scenarios=request.scenarios,
        seed=request.seed,
    )
    konfig = ADMMConfig(
        verbose=False,
        solver_name="gurobi",
        solver_non_convex=request.config.solver_non_convex,
        big_m=request.config.big_m,
        ε=request.config.ε,
        ρ=request.config.ρ,
        γ_infeasibility=request.config.γ_infeasibility,
        γ_admm_penalty=request.config.γ_admm_penalty,
        γ_trafo_loss=request.config.γ_trafo_loss,
        time_limit=request.config.time_limit,
        groups=request.config.groups,
        max_iters=request.config.max_iters,
        μ=request.config.μ,
        τ_incr=request.config.τ_incr,
        τ_decr=request.config.τ_decr,
    )
    dap = DigAPlanADMM(konfig=konfig)
    dap.add_grid_data(base_grid_data)
    dap.solve_model(groups=request.config.groups)

    switches = dap.model_manager.zδ_variable
    taps = dap.model_manager.zζ_variable
    voltages = dap.result_manager.extract_node_voltage(scenario=0)
    currents = dap.result_manager.extract_edge_current(scenario=0)

    if request.save_path:
        save_dap_state(dap, request.save_path)

    return ReconfigurationOutput(
        switches=switches.to_dicts(),
        voltages=voltages.to_dicts(),
        currents=currents.to_dicts(),
        taps=taps.to_dicts(),
    )
