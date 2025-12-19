from api import (
    GridCase,
    GridCaseModel,
    ReconfigurationOutput,
)
from api.grid_cases import get_grid_case
from experiments import *


class ADMMInput(GridCaseModel):
    groups: int | dict[int, list[int]] = 10
    max_iters: int = 10


class ADMMOutput(ReconfigurationOutput):
    pass


def run_admm(input: ADMMInput) -> ADMMOutput:
    net, base_grid_data = get_grid_case(input)
    config = ADMMConfig(
        verbose=False,
        pipeline_type=PipelineType.ADMM,
        solver_name="gurobi",
        solver_non_convex=2,
        big_m=1e3,
        ε=1 if input.grid_case != GridCase.SIMPLE_GRID else 1e-4,
        ρ=2.0,
        γ_infeasibility=10 if input.grid_case == GridCase.SIMPLE_GRID else 100.0,
        γ_admm_penalty=1.0,
        γ_trafo_loss=1e2 if input.grid_case == GridCase.SIMPLE_GRID else 1.0,
        time_limit=1 if input.grid_case == GridCase.BOISY_SIMPLIFIED else 10,
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

    return ADMMOutput(
        switches=switches.to_dicts(),
        voltages=voltages.to_dicts(),
        currents=currents.to_dicts(),
        taps=taps.to_dicts(),
    )
