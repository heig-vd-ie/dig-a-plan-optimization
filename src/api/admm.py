from data_model.reconfiguration import ADMMInput, ReconfigurationOutput
from api.grid_cases import get_grid_case
from experiments import *


def run_admm(input: ADMMInput) -> ReconfigurationOutput:
    net, base_grid_data = get_grid_case(
        grid=input.grid, seed=input.seed, stu=input.scenarios
    )
    config = ADMMConfig(
        verbose=False,
        solver_name="gurobi",
        solver_non_convex=2,
        big_m=1e3,
        ε=1 if input.grid.pp_file != "examples/ieee-33/simple_grid.p" else 1e-4,
        ρ=2.0,
        γ_infeasibility=(
            10 if input.grid.pp_file == "examples/ieee-33/simple_grid.p" else 100.0
        ),
        γ_admm_penalty=1.0,
        γ_trafo_loss=(
            1e2 if input.grid.pp_file == "examples/ieee-33/simple_grid.p" else 1.0
        ),
        time_limit=(1 if input.grid.pp_file == "examples/boisy_simplified.p" else 10),
        groups=input.groups,
        max_iters=input.max_iters,
        μ=10.0,
        τ_incr=2.0,
        τ_decr=2.0,
    )
    dap = DigAPlanADMM(config=config)
    dap.add_grid_data(base_grid_data)
    dap.solve_model(groups=input.groups)
    # Fixed switches solution (for distribution plots)
    dap_fixed = copy.deepcopy(dap)
    dap_fixed.solve_model(fixed_switches=True)

    switches = dap.model_manager.zδ_variable
    taps = dap.model_manager.zζ_variable
    voltages = dap.result_manager.extract_node_voltage(scenario=0)
    currents = dap.result_manager.extract_edge_current(scenario=0)

    save_dap_state(dap, ".cache/figs/boisy_dap")
    save_dap_state(dap_fixed, ".cache/figs/boisy_dap_fixed")
    joblib.dump(net, ".cache/figs/boisy_net.joblib")

    return ReconfigurationOutput(
        switches=switches.to_dicts(),
        voltages=voltages.to_dicts(),
        currents=currents.to_dicts(),
        taps=taps.to_dicts(),
    )
