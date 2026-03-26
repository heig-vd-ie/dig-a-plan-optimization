from pathlib import Path
import pandapower as pp
import polars as pl
import pandas as pd
import ray
from typing import Dict
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
        self.benchmark_expansion = benchmark_expansion
        self.number_of_quarters = 400  # TODO
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

        for year in [benchmark_expansion.congestion_settings.years[0]]:  # TODO
            for quarter in range(1, 1 + 2):  # TODO
                log.info(f"Power flow for year {year} and quarter {quarter}")
                results = self.run_per_year(benchmark_expansion, year, quarter)
                self.reinforce_edges(results, year, quarter)
        return None

    def reinforce_edges(
        self, results: Dict[str, PowerFlowResponse], year: int, quarter: int
    ):
        # TODO: voltage dependent expansion
        frames_lines: list[pd.DataFrame] = []
        frames_trafo: list[pd.DataFrame] = []

        for i in list(results.keys()):

            df = pd.DataFrame(results[i].congested_lines)
            frames_lines.append(df)

            df = pd.DataFrame(results[i].congested_trafos)
            frames_trafo.append(df)

        congested_lines = pd.concat(frames_lines)
        congested_trafo = pd.concat(frames_trafo)

        if congested_lines.shape[0] > 0:
            congested_lines = congested_lines.groupby("line_idx").max()
        else:
            congested_lines = pd.DataFrame(columns=["line_idx", "loading_percent"])
        if congested_trafo.shape[0] > 0:
            congested_trafo = congested_trafo.groupby("trafo_idx").max()
        else:
            congested_trafo = pd.DataFrame(columns=["trafo_idx", "loading_percent"])

        log.info(congested_lines)
        log.info(congested_trafo)
        self.net0.line = self.net0.line.merge(
            congested_lines, left_on="idx", right_on="line_idx"
        )
        self.net0.trafo = self.net0.trafo.merge(
            congested_trafo, left_on="idx", right_on="trafo_idx"
        )
        self.net0.line["loading_percent"] = self.net0.line["loading_percent"].fillna(
            self.benchmark_expansion.congestion_settings.threshold
        )
        self.net0.line["new_ka"] = self.net0.line["max_i_ka"] * (
            self.net0.line["loading_percent"]
            / self.benchmark_expansion.congestion_settings.threshold
        )
        df = self.net0.line.copy()
        df["delta_ka"] = self.net0.line["new_ka"] - self.net0.line["max_i_ka"]
        df = df[df["delta_ka"] > 0]
        save_obj_to_json(
            obj=df[["delta_ka"]].to_dict(),
            path_filename=PROJECT_ROOT
            / settings.cache.outputs_benchmark
            / self.benchmark_expansion.grid.name
            / f"expanded_lines_{self.benchmark_expansion.profiles.scenario_name}_{year}_{quarter}.json",
        )
        self.net0.line["max_i_ka"] = self.net0.line["new_ka"]
        self.net0.line = self.net0.line.drop(columns=["loading_percent", "new_ka"])

        self.net0.trafo["loading_percent"] = self.net0.trafo["loading_percent"].fillna(
            self.benchmark_expansion.congestion_settings.threshold
        )
        self.net0.trafo["new_mva"] = self.net0.trafo["sn_mva"] * (
            self.net0.trafo["loading_percent"]
            / self.benchmark_expansion.congestion_settings.threshold
        )
        df = self.net0.trafo.copy()
        df["delta_mva"] = self.net0.trafo["new_mva"] - self.net0.trafo["sn_mva"]
        df = df[df["delta_mva"] > 0]
        save_obj_to_json(
            obj=df[["delta_mva"]].to_dict(),
            path_filename=PROJECT_ROOT
            / settings.cache.outputs_benchmark
            / self.benchmark_expansion.grid.name
            / f"expanded_trafos_{self.benchmark_expansion.profiles.scenario_name}_{year}_{quarter}.json",
        )
        self.net0.trafo["sn_mva"] = self.net0.trafo["new_mva"]
        self.net0.trafo = self.net0.trafo.drop(columns=["loading_percent", "new_mva"])

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
