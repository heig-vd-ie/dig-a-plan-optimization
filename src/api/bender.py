from pydantic import BaseModel
from api.models import GridCaseModel, ReconfigurationOutput, ShortTermUncertainty
from experiments import *
from api.grid_cases import get_grid_case


class BenderInput(BaseModel):
    grid: GridCaseModel = GridCaseModel()
    max_iters: int = 100
    scenarios: ShortTermUncertainty = ShortTermUncertainty()
    seed: int = 42


class BenderOutput(ReconfigurationOutput):
    pass


def run_bender(input: BenderInput) -> BenderOutput:
    net, base_grid_data = get_grid_case(
        grid=input.grid, seed=input.seed, stu=input.scenarios
    )
    config = BenderConfig(
        verbose=False,
        big_m=1e2,
        factor_p=1e-3,
        factor_q=1e-3,
        factor_v=1,
        factor_i=1e-3,
        master_relaxed=False,
        pipeline_type=PipelineType.BENDER,
    )
    dig_a_plan = DigAPlanBender(config=config)
    dig_a_plan.add_grid_data(base_grid_data)
    dig_a_plan.solve_model(max_iters=input.max_iters)
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(
        go.Scatter(
            go.Scatter(y=dig_a_plan.model_manager.slave_obj_list[1:]),
            mode="lines",
            name="Slave objective",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            go.Scatter(y=dig_a_plan.model_manager.master_obj_list[1:]),
            mode="lines",
            name="Master objective",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            go.Scatter(y=dig_a_plan.model_manager.convergence_list[1:]),
            mode="lines",
            line=dict(dash="dot"),
            name="Difference",
        ),
        row=1,
        col=1,
    )
    fig.update_layout(
        height=600,
        width=1200,
        margin=dict(t=10, l=20, r=10, b=10),
        legend=dict(
            x=0.70,  # Position legend inside the plot area
            y=0.98,  # Position at top-left
            bgcolor="rgba(255,255,255,0.8)",  # Semi-transparent white background
            bordercolor="rgba(0,0,0,0.2)",  # Light border
            borderwidth=1,
        ),
        xaxis_title="Iteration",
        yaxis_title="Objective Value",
    )
    os.makedirs(".cache/figs", exist_ok=True)
    fig.write_html(".cache/figs/bender-convergence.html")
    plot_grid_from_pandapower(net=net, dap=dig_a_plan)
    plot_grid_from_pandapower(net=net, dap=dig_a_plan, color_by_results=True)
    print(
        extract_optimization_results(
            dig_a_plan.model_manager.master_model_instance, "δ"
        )
        .to_pandas()
        .to_string()
    )
    switches = dig_a_plan.result_manager.extract_switch_status()
    voltages = dig_a_plan.result_manager.extract_node_voltage()
    currents = dig_a_plan.result_manager.extract_edge_current()
    taps = dig_a_plan.result_manager.extract_transformer_tap_position()
    return BenderOutput(
        switches=switches.to_dicts(),
        voltages=voltages.to_dicts(),
        currents=currents.to_dicts(),
        taps=taps.to_dicts(),
    )
