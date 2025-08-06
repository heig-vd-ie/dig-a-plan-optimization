import polars as pl
import pandapower as pp
from data_display.output_processing import compare_dig_a_plan_with_pandapower
from data_exporter.pandapower_to_dig_a_plan import pandapower_to_dig_a_plan_schema
from pipelines import DigAPlanADMM, DigAPlanCombined
from pipelines.configs import ADMMConfig, CombinedConfig, PipelineType


def test_admm_model_simple_example():

    net = pp.from_pickle("data/simple_grid.p")

    groups = {
        0: [19, 20, 21, 29, 32, 35],
        1: [35, 30, 33, 25, 26, 27],
        2: [27, 32, 22, 23, 34],
        3: [31, 24, 28, 21, 22, 23],
        4: [34, 26, 25, 24, 31],
    }

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
        max_iters=10,
        μ=10.0,
        τ_incr=2.0,
        τ_decr=2.0,
        groups=groups,
    )

    dap = DigAPlanADMM(config=config)

    dap.add_grid_data(grid_data)

    switch_ids = dap.data_manager.edge_data.filter(pl.col("type") == "switch")[
        "edge_id"
    ].to_list()
    scen_ids = list(grid_data.load_data.keys())
    print("Switch IDs:", switch_ids)
    print("Scenario IDs:", scen_ids)

    dap.model_manager.solve_model()

    print("\n--- ADMM consensus z per switch ---")
    print(dap.model_manager.z)  # {switch_id: z}

    print("\n--- ADMM last-iterate delta (per scenario, per switch) ---")
    print(dap.model_manager.δ_variable)  # Polars DF: ["SCEN","S","δ_variable"]

    switches1 = dap.data_manager.edge_data.filter(pl.col("type") == "switch").select(
        "eq_fk", "edge_id", "normal_open"
    )

    z_df = pl.DataFrame(
        {
            "edge_id": list(dap.model_manager.z.keys()),
            "z": list(dap.model_manager.z.values()),
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
    dig_a_plan = DigAPlanCombined(config=config)

    dig_a_plan.add_grid_data(grid_data)
    dig_a_plan.solve_model()  # one‐shot solve

    # Switch status
    switches2 = dig_a_plan.result_manager.extract_switch_status()

    assert (
        consensus_states.select(["eq_fk", "open"])
        .to_pandas()
        .equals(switches2.select(["eq_fk", "open"]).to_pandas())
    )
