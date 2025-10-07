import random
import numpy as np
from typing import Dict
import patito as pt
import polars as pl
import pandapower as pp
import logging
import tqdm
from config import settings
from data_schema import LoadData
from general_function import duckdb_to_dict
from power_profiles.download_from_swisstopo import download_estavayer_power_profiles
from power_profiles.models import LoadType

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
QF_RATED = 0.1
KW_TO_W = 1e3


class ScenarioFactory:

    def __init__(self, kace: str):
        self.kace = kace
        self.dfs: Dict[str, pl.DataFrame] = {}
        self.load_duckdb: Dict[str, pl.DataFrame] = {}
        self.net: pp.pandapowerNet | None = None
        self.rand_scenarios: Dict[int, pt.DataFrame[LoadData]] = {}

    def initialize(self):
        if self.kace not in settings.cases:
            raise ValueError(f"Unknown kace: {self.kace}")
        download_estavayer_power_profiles(kace=self.kace, force=False)
        self.dfs = {
            k.value: pl.read_parquet(
                f"{settings.cases[self.kace].load_allocation_folder}/{k.value}.parquet"
            )
            for k in LoadType
        }
        self.load_duckdb = duckdb_to_dict(settings.cases[self.kace].load_duckdb_file)
        self.net = pp.from_pickle(settings.cases[self.kace].pandapower_file)
        return self

    def generate_operational_scenarios(
        self,
        number_of_random_scenarios: int = 10,
        s_base: float = 1e6,
        v_bounds: tuple[float, float] | None = None,
        seed: int = 42,
    ):
        if self.net is None or not self.dfs or not self.load_duckdb:
            raise ValueError(
                "ScenarioFactory not initialized. Call initialize() first."
            )
        if v_bounds is None:
            v_bounds = (-0.1, 0.1)
        random.seed(seed)
        rng = np.random.default_rng(seed)
        total_load = (
            self.dfs[LoadType.TOTAL.value]
            .select(pl.all().shuffle(seed=seed))
            .head(number_of_random_scenarios)
        )
        pv_load = self.dfs[LoadType.PV.value].filter(
            pl.col("timestamp").is_in(total_load.get_column("timestamp").to_list())
        )
        egid_columns = list(
            set(
                [col for col in total_load.columns if col != "timestamp"]
                + [col for col in pv_load.columns if col != "timestamp"]
            )
        )

        non_detected_egids = []
        for idx, row in tqdm.tqdm(
            enumerate(total_load.iter_rows(named=True)),
            desc="Generating scenarios",
            total=number_of_random_scenarios,
        ):
            timestamp = row["timestamp"]
            load_data_list = []
            for node_id in self.net.bus.index:
                load_data_list.append(
                    LoadData(
                        node_id=node_id,
                        p_cons_pu=0.0,
                        q_cons_pu=0.0,
                        p_prod_pu=0.0,
                        q_prod_pu=0.0,
                        v_node_sqr_pu=rng.uniform(
                            (1 + v_bounds[0]) ** 2, (1 + v_bounds[1]) ** 2
                        ),
                    )
                )
            for egid in egid_columns:
                node_id = list(
                    set(self.net.load[self.net.load["egid"] == egid]["bus"].to_list())
                )
                if not node_id:
                    if egid not in non_detected_egids:
                        logger.warning(f"Warning: No node found for egid {egid}")
                        non_detected_egids.append(egid)
                    continue
                node_id = node_id[0]
                p_prod_pu = abs(
                    pv_load.filter(pl.col("timestamp") == timestamp)
                    .get_column(egid)
                    .to_list()[0]
                    / s_base
                    if egid in pv_load.columns
                    else 0.0
                )
                q_prod_pu = p_prod_pu * KW_TO_W * QF_RATED

                p_cons_pu = max([0, row.get(egid, 0.0) * KW_TO_W / s_base + p_prod_pu])
                q_cons_pu = p_cons_pu * QF_RATED

                v_node_sqr_pu = rng.uniform(
                    (1 + v_bounds[0]) ** 2, (1 + v_bounds[1]) ** 2
                )
                load_data_list.append(
                    LoadData(
                        node_id=node_id,
                        p_cons_pu=p_cons_pu,
                        q_cons_pu=q_cons_pu,
                        p_prod_pu=p_prod_pu,
                        q_prod_pu=q_prod_pu,
                        v_node_sqr_pu=v_node_sqr_pu,
                    )
                )
            load_df = (
                pl.DataFrame(load_data_list)
                .group_by("node_id")
                .agg(
                    [
                        pl.col("p_cons_pu").sum().alias("p_cons_pu"),
                        pl.col("q_cons_pu").sum().alias("q_cons_pu"),
                        pl.col("p_prod_pu").sum().alias("p_prod_pu"),
                        pl.col("q_prod_pu").sum().alias("q_prod_pu"),
                        pl.col("v_node_sqr_pu").mean().alias("v_node_sqr_pu"),
                    ]
                )
            ).with_columns(
                pl.col("node_id").cast(pl.Int32),
            )
            self.rand_scenarios[idx] = pt.DataFrame(load_df).set_model(LoadData)
            self.rand_scenarios[idx].validate()
        return self


if __name__ == "__main__":
    scenarios = (
        ScenarioFactory(kace="estavayer_centre_ville")
        .initialize()
        .generate_operational_scenarios()
    )

    print(scenarios.rand_scenarios[1].sum())
