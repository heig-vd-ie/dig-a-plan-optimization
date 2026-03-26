if __name__ == "__main__":
    from pathlib import Path
    from data_model.benchmark import BenchmarkExpansion
    import pandapower as pp
    import polars as pl
    from helpers.congestion import heavy_task_powerflow
    import json

    with open(
        Path(__file__).parents[2]
        / "experiments"
        / "expansion_benchmark"
        / "00-settings.json"
    ) as f:
        payload = json.load(f)

    benchmark_expansion = BenchmarkExpansion(**payload)

    net0 = pp.from_pickle(Path(__file__).parents[2] / benchmark_expansion.grid.pp_file)
    map_df = pl.read_csv(
        Path(__file__).parents[2] / benchmark_expansion.grid.egid_id_mapping_file
    ).select([pl.col("index"), pl.col("egid").cast(pl.Utf8)])

    net0.sgen = net0.load.copy()
    net0.sgen["p_mw"] = 0
    net0.sgen["q_mvar"] = 0

    year = benchmark_expansion.congestion_settings.years[0]
    load_dfs: list[pl.DataFrame] = []
    for f in benchmark_expansion.profiles.load_profiles:
        load_parquet_file = (
            Path(__file__).parents[2]
            / f
            / f"{benchmark_expansion.profiles.scenario_name.value}_{year}.parquet"
        )
        load_dfs.append(pl.read_parquet(load_parquet_file))
    load_profiles_all_egids = pl.concat(load_dfs).join(map_df, on="egid", how="left")
    pv_parquet_file = (
        Path(__file__).parents[2]
        / benchmark_expansion.profiles.pv_profile
        / f"{benchmark_expansion.profiles.scenario_name.value}_{year}.parquet"
    )
    pv_profiles_all_egids = pl.read_parquet(pv_parquet_file).join(
        map_df, on="egid", how="left"
    )
    heavy_task_powerflow(
        net0,
        benchmark_expansion.grid.name,
        load_profiles_all_egids,
        pv_profiles_all_egids,
        "_0",
        benchmark_expansion.grid.cosφ,
        year,
        benchmark_expansion.profiles.scenario_name.value,
    )
