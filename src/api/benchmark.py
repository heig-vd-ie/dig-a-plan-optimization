from pathlib import Path
import pandapower as pp
import polars as pl
import json

import ray

from api.ray_utils import init_ray, shutdown_ray, check_ray
from data_model.benchmark_expansion import BenchmarkExpansion
from helpers.congestion import heavy_task_powerflow
from tqdm import tqdm
from helpers import generate_log

log = generate_log(name=__name__)


def run_benchmark(benchmark_expansion: BenchmarkExpansion):
    """"""
    net0 = pp.from_pickle(Path(__file__).parents[2] / benchmark_expansion.grid.pp_file)
    map_df = pl.read_csv(
        Path(__file__).parents[2] / benchmark_expansion.grid.egid_id_mapping_file
    ).select([pl.col("index"), pl.col("egid").cast(pl.Utf8)])
    load_profile_dir = (
        Path(__file__).parents[2] / benchmark_expansion.profiles.load_profiles[0]
    )
    pv_profile_dir = Path(__file__).parents[2] / benchmark_expansion.profiles.pv_profile

    net0.sgen = net0.load
    net0.sgen["p_mw"] = 0
    net0.sgen["q_mvar"] = 0

    for year in benchmark_expansion.congestion_settings.years:
        log.info(f"Power flow for year {year}")
        load_parquet_file = (
            load_profile_dir
            / f"{benchmark_expansion.profiles.scenario_name.value}_{year}.parquet"
        )
        pv_parquet_file = (
            pv_profile_dir
            / f"{benchmark_expansion.profiles.scenario_name.value}_{year}.parquet"
        )
        load_profiles_all_egids = pl.read_parquet(load_parquet_file).join(
            map_df, on="egid", how="left"
        )
        pv_profiles_all_egids = pl.read_parquet(pv_parquet_file).join(
            map_df, on="egid", how="left"
        )
        time_cols = [c for c in load_profiles_all_egids.columns if c.startswith("_")]

        init_ray()
        check_ray(True)
        heavy_task_remote = ray.remote(heavy_task_powerflow)
        net0_ref = ray.put(net0)
        load_df_ref = ray.put(load_profiles_all_egids)
        pv_df_ref = ray.put(pv_profiles_all_egids)
        futures = {
            t: heavy_task_remote.remote(
                net0_ref,
                load_df_ref,
                pv_df_ref,
                t,
                benchmark_expansion.grid.cosφ,
                benchmark_expansion.congestion_settings.threshold,
            )
            for t in time_cols
        }
        future_results = {}
        for t in tqdm(time_cols, desc="Running PowerFlow"):
            try:
                future_results[t] = ray.get(futures[t])
            except:
                log.error(f"ERROR in [{t}]!!!")
        shutdown_ray()
    return None
