import polars as pl
import pandapower as pp
from data_display.output_processing import compare_dig_a_plan_with_pandapower
from data_exporter.pandapower_to_dig_a_plan import pandapower_to_dig_a_plan_schema
from pipelines import DigAPlan
from pipelines.configs import ADMMConfig, CombinedConfig, PipelineType
from pipelines.model_managers.admm import PipelineModelManagerADMM
from pipelines.model_managers.admm import PipelineModelManagerADMM


def test_admm_model_simple_example():
    LOAD_FACTOR = 1
    TEST_CONFIG = [
        {"line_list": [], "switch_list": []},
        {"line_list": [6, 9], "switch_list": [25, 28]},
        {"line_list": [2, 6, 9], "switch_list": [21, 25, 28]},
        {"line_list": [16], "switch_list": [35]},
        {"line_list": [1], "switch_list": [20]},
        {"line_list": [10], "switch_list": [29]},
        {"line_list": [7, 11], "switch_list": [26, 30]},
    ]
    NB_TEST = 0

    net = pp.from_pickle("data/simple_grid.p")

    net["load"]["p_mw"] = net["load"]["p_mw"] * LOAD_FACTOR
    net["load"]["q_mvar"] = net["load"]["q_mvar"] * LOAD_FACTOR

    net["line"].loc[:, "max_i_ka"] = 1
    net["line"].loc[TEST_CONFIG[NB_TEST]["line_list"], "max_i_ka"] = 1e-2

    LOAD_FACTOR = 1.0
    net["load"]["p_mw"] *= LOAD_FACTOR
    net["load"]["q_mvar"] *= LOAD_FACTOR
    net["line"].loc[:, "max_i_ka"] = 1.0

    grid_data = pandapower_to_dig_a_plan_schema(net)

    config = ADMMConfig(
        verbose=False,
        pipeline_type=PipelineType.ADMM,
        solver_name="gurobi",
        solver_non_convex=2,
        big_m=1e3,
        ε=1e-4,
        ρ=2.0,  # initial rho
        γ_infeasibility=1.0,
        γ_admm_penalty=1.0,
    )

    dap = DigAPlan(config=config)

    dap.add_grid_data(grid_data)

    # Sanity on manager type
    assert isinstance(
        dap.model_manager, PipelineModelManagerADMM
    ), "PipelineType must be ADMM."

    switch_ids = dap.data_manager.edge_data.filter(pl.col("type") == "switch")[
        "edge_id"
    ].to_list()
    scen_ids = list(grid_data.load_data.keys())
    print("Switch IDs:", switch_ids)
    print("Scenario IDs:", scen_ids)

    dap.model_manager.solve_model(
        max_iters=10,
        μ=10.0,
        τ_incr=2.0,
        τ_decr=2.0,
    )

    print("\n--- ADMM consensus z per switch ---")
    print(dap.model_manager.admm_z)  # {switch_id: z}

    print("\n--- ADMM last-iterate delta (per scenario, per switch) ---")
    print(dap.model_manager.δ_variable)  # Polars DF: ["SCEN","S","δ_variable"]

    switches1 = dap.data_manager.edge_data.filter(pl.col("type") == "switch").select(
        "eq_fk", "edge_id", "normal_open"
    )

    z_df = pl.DataFrame(
        {
            "edge_id": list(dap.model_manager.admm_z.keys()),
            "z": list(dap.model_manager.admm_z.values()),
        }
    )

    consensus_states = (
        switches1.join(z_df, on="edge_id", how="inner")
        .with_columns(
            (pl.col("z") > 0.5).alias("closed"), (~(pl.col("z") > 0.5)).alias("open")
        )
        .select("eq_fk", "edge_id", "z", "normal_open", "closed", "open")
        .sort("edge_id")
    )

    print("\n=== ADMM consensus switch states (z) ===")
    print(consensus_states)

    node_data, edge_data = compare_dig_a_plan_with_pandapower(dig_a_plan=dap, net=net)
    assert node_data.get_column("v_diff").abs().max() < 1e-3  # type: ignore
    assert edge_data.get_column("i_diff").abs().max() < 0.1  # type: ignore

    config = CombinedConfig(
        verbose=True,
        big_m=1e3,
        ε=1,
        pipeline_type=PipelineType.COMBINED,
        γ_infeasibility=1.0,
        γ_admm_penalty=0.0,
        all_scenarios=True,
    )
    dig_a_plan = DigAPlan(config=config)

    dig_a_plan.add_grid_data(grid_data)
    dig_a_plan.solve_model()  # one‐shot solve

    # Switch status
    switches2 = dig_a_plan.result_manager.extract_switch_status()

    assert (
        consensus_states.select(["eq_fk", "open"])
        .to_pandas()
        .equals(switches2.select(["eq_fk", "open"]).to_pandas())
    )
