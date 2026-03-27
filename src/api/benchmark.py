from pathlib import Path
import pandapower as pp
import polars as pl
import pandas as pd
import ray
import os
from typing import Dict, Literal
from api.ray_utils import init_ray, shutdown_ray, check_ray
from data_model.benchmark import BenchmarkExpansion, PowerFlowResponse
from helpers.congestion import heavy_task_powerflow
from tqdm import tqdm
from helpers import generate_log
from helpers.json import save_obj_to_json
from konfig import settings, PROJECT_ROOT

log = generate_log(name=__name__)


class Benchmark:

    def run(self, benchmark_expansion: BenchmarkExpansion):
        """"""
        if not (
            PROJECT_ROOT
            / settings.cache.outputs_benchmark
            / benchmark_expansion.grid.name
        ).exists():
            os.makedirs(
                str(
                    PROJECT_ROOT
                    / settings.cache.outputs_benchmark
                    / benchmark_expansion.grid.name
                ),
                exist_ok=True,
            )
        save_obj_to_json(
            obj=benchmark_expansion,
            path_filename=PROJECT_ROOT
            / settings.cache.outputs_benchmark
            / benchmark_expansion.grid.name
            / f"input.json",
        )
        self.benchmark_expansion = benchmark_expansion
        self.number_of_quarters = 4
        self.net0 = pp.from_pickle(
            Path(__file__).parents[2] / benchmark_expansion.grid.pp_file
        )
        self.net0.line["idx"] = self.net0.line.index
        self.net0.trafo["idx"] = self.net0.trafo.index
        self.map_df = pl.read_csv(
            Path(__file__).parents[2] / benchmark_expansion.grid.egid_id_mapping_file
        ).select([pl.col("index"), pl.col("egid").cast(pl.Utf8)])

        self.net0.sgen = self.net0.load.copy()
        self.net0.sgen["p_mw"] = 0
        self.net0.sgen["q_mvar"] = 0

        for year in benchmark_expansion.congestion_settings.years:
            for quarter in range(1, 1 + self.number_of_quarters):
                log.info(f"Power flow for year {year} and quarter {quarter}")
                results = self.run_per_year(benchmark_expansion, year, quarter)
                self.reinforce_edges(results, year, quarter, "line")
                self.reinforce_edges(results, year, quarter, "trafo")
        return None

    def reinforce_edges(
        self,
        results: Dict[str, PowerFlowResponse],
        year: int,
        quarter: int,
        edge_type: Literal["line", "trafo"],
    ):
        """Reinforce each edge based on loading percent and voltage"""
        capacity_column = "max_i_ka" if edge_type == "line" else "sn_mva"
        frames_i: list[pd.DataFrame] = []
        frames_v: list[pd.DataFrame] = []
        for i in list(results.keys()):
            df1 = pd.DataFrame(getattr(results[i], f"congested_{edge_type}s"))
            df2 = pd.DataFrame(results[i].congested_buses)
            frames_i.append(df1)
            frames_v.append(df2)

        congested = pd.concat(frames_i)

        if congested.shape[0] > 0:
            congested = congested.groupby(f"{edge_type}_idx").max()
        else:
            congested = pd.DataFrame(columns=[f"{edge_type}_idx", "loading_percent"])

        log.info(congested)
        new_edges = getattr(self.net0, edge_type).copy()
        new_edges = new_edges.merge(
            congested, left_on="idx", right_on=f"{edge_type}_idx"
        )
        new_edges["loading_percent"] = new_edges["loading_percent"].fillna(
            self.benchmark_expansion.congestion_settings.threshold
        )

        new_edges["new_cap"] = new_edges[capacity_column] * (
            new_edges["loading_percent"]
            / self.benchmark_expansion.congestion_settings.threshold
        )
        new_edges["delta"] = new_edges["new_cap"] - new_edges[capacity_column]
        new_edges = new_edges[new_edges["delta"] > 0]
        save_obj_to_json(
            obj=new_edges[["delta"]].to_dict(),
            path_filename=PROJECT_ROOT
            / settings.cache.outputs_benchmark
            / self.benchmark_expansion.grid.name
            / f"expanded_{edge_type}s_{self.benchmark_expansion.profiles.scenario_name}_{year}_{quarter}.json",
        )
        new_edges[capacity_column] = new_edges["new_cap"]
        new_edges = new_edges.drop(columns=["loading_percent", "new_cap", "delta"])
        setattr(self.net0, edge_type, new_edges)

    def run_per_year(
        self, benchmark_expansion: BenchmarkExpansion, year: int, quarter: int
    ) -> Dict[str, PowerFlowResponse]:
        load_dfs: list[pl.DataFrame] = []
        for f in benchmark_expansion.profiles.load_profiles:
            load_parquet_file = (
                Path(__file__).parents[2]
                / f
                / f"{benchmark_expansion.profiles.scenario_name.value}_{year}.parquet"
            )
            load_dfs.append(pl.read_parquet(load_parquet_file))
        load_profiles_all_egids = (
            pl.concat(load_dfs)
            .select(
                ["egid"]
                + [
                    f"_{i}"
                    for i in range(
                        int(8760 / self.number_of_quarters) * (quarter - 1),
                        int(8760 / self.number_of_quarters) * quarter,
                    )
                ]
            )
            .join(self.map_df, on="egid", how="left")
        )
        pv_parquet_file = (
            Path(__file__).parents[2]
            / benchmark_expansion.profiles.pv_profile
            / f"{benchmark_expansion.profiles.scenario_name.value}_{year}.parquet"
        )
        pv_profiles_all_egids = (
            pl.read_parquet(pv_parquet_file)
            .select(
                ["egid"]
                + [
                    f"_{i}"
                    for i in range(
                        int(8760 / self.number_of_quarters) * (quarter - 1),
                        int(8760 / self.number_of_quarters) * quarter,
                    )
                ]
            )
            .join(self.map_df, on="egid", how="left")
        )
        time_cols = [c for c in load_profiles_all_egids.columns if c.startswith("_")]

        init_ray()
        check_ray(True)
        heavy_task_remote = ray.remote(heavy_task_powerflow)
        net0_ref = ray.put(self.net0)
        load_df_ref = ray.put(load_profiles_all_egids)
        pv_df_ref = ray.put(pv_profiles_all_egids)
        futures = {
            t: heavy_task_remote.remote(
                net0_ref,
                benchmark_expansion.grid.name,
                load_df_ref,
                pv_df_ref,
                t,
                benchmark_expansion.grid.cosφ,
                year,
                benchmark_expansion.profiles.scenario_name.value,
                benchmark_expansion.congestion_settings.threshold,
                benchmark_expansion.congestion_settings.voltage_limits,
            )
            for t in time_cols
        }
        future_results: Dict[str, PowerFlowResponse] = {}
        for t in tqdm(time_cols, desc="Running PowerFlow"):
            try:
                future_results[t] = ray.get(futures[t])
            except:
                log.error(f"ERROR in [{t}]!!!")
        shutdown_ray()
        return future_results
