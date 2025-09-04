from api import *


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
    print(dap.model_manager.zδ_variable)
    print(dap.model_manager.zζ_variable)
    if input.grid_case == GridCase.SIMPLE_GRID:
        node_data, edge_data = compare_dig_a_plan_with_pandapower(
            dig_a_plan=dap, net=net, from_z=True
        )
        fig = plot_grid_from_pandapower(net=net, dap=dap, from_z=True)

    dap_fixed = copy.deepcopy(dap)
    dap_fixed.solve_model(fixed_switches=True)

    nodal_variables = [
        "voltage",
        # "p_curt_cons",
        # "p_curt_prod",
        # "q_curt_cons",
        # "q_curt_prod",
    ]
    edge_variables = [
        "current",
        "p_flow",
        "q_flow",
    ]
    for variable in nodal_variables + edge_variables:
        plot_distribution_variable(
            daps={"ADMM": dap, "Normal Open": dap_fixed},
            variable_name=variable,
            variable_type=("nodal" if variable in nodal_variables else "edge"),
        )

    os.makedirs(".cache/figs", exist_ok=True)
    x_vals = np.array(dap.model_manager.time_list[1:]) - dap.model_manager.time_list[0]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=dap.model_manager.r_norm_list,
            mode="lines+markers",
            name="r_norm",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=dap.model_manager.s_norm_list,
            mode="lines+markers",
            name="s_norm",
        )
    )
    fig.update_layout(
        title="ADMM Iteration: r_norm and s_norm",
        xaxis_title="Seconds",
        yaxis_title="Norm Value",
        legend=dict(x=0.01, y=0.99),
        template="plotly_white",
        width=1000,
        height=500,
    )
    fig.write_html(".cache/figs/admm_iterations.html")

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
