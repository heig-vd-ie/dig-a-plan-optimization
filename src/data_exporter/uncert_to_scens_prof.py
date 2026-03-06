from abc import ABC, abstractmethod
from typing import Dict
from pathlib import Path
import random
import numpy as np
import polars as pl
import numpy as np
import pandapower as pp
import patito as pt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from data_model import GridCaseModel, ShortTermUncertaintyProfile, LoadData
from data_model.kace import (
    DiscreteScenario,
)
from helpers import generate_log

log = generate_log(__name__)


class ReductionStrategy(ABC):
    """
    Returns INDICES of the representative scenarios, not the values.
    This allows us to map back to the original Polars DataFrame.
    """

    @abstractmethod
    def get_representative_indices(
        self, feature_matrix: np.ndarray, n_scenarios: int, seed: int = 42
    ) -> np.ndarray:
        """
        Returns:
            indices: Array of integer indices pointing to the representative rows.
        """
        pass


class KMeansMedoidReducer(ReductionStrategy):
    def get_representative_indices(
        self, feature_matrix: np.ndarray, n_scenarios: int, seed: int = 42
    ) -> np.ndarray:
        # Fit KMeans
        scaler = StandardScaler()
        feature_matrix_scaled = scaler.fit_transform(feature_matrix)
        kmeans = KMeans(n_clusters=n_scenarios, random_state=seed, n_init=10)
        kmeans.fit(feature_matrix_scaled)
        # Find the ACTUAL data point closest to each cluster center
        # 'closest_row_indices' maps cluster_id -> row_index in feature_matrix
        closest_row_indices, _ = pairwise_distances_argmin_min(
            kmeans.cluster_centers_, feature_matrix_scaled
        )
        return closest_row_indices


class ScenarioPipelineProfile:
    def __init__(self, reduction_strategy: ReductionStrategy = KMeansMedoidReducer()):
        self.reducer = reduction_strategy

    def process(self, ksop: ShortTermUncertaintyProfile):
        """
        1. Filters Polars DF by Year/Scenario
        2. Extracts features
        3. Gets representative indices
        4. Returns reduced Polars DF with weights
        """
        # A. Filter Data (Polars Operation)
        # Assuming input has columns like 'year', 'scenario_type', 'quarter'
        self.ksop = ksop
        try:
            load_dfs = []
            for load_profile_path in self.ksop.load_profiles:
                df = pl.read_parquet(
                    Path(load_profile_path)
                    / f"{self.ksop.scenario_name.value}_{self.ksop.target_year}.parquet"
                )
                load_dfs.append(df)
            pv_df = pl.read_parquet(
                Path(self.ksop.pv_profile)
                / f"{self.ksop.scenario_name.value}_{self.ksop.target_year}.parquet"
            )
        except Exception as e:
            raise ValueError(f"Error reading profile data: {e}")

        # B. Feature Extraction (Polars -> Numpy)
        load_df: pl.DataFrame = pl.concat(load_dfs, how="vertical")
        df = pl.concat([load_df, pv_df], how="vertical").drop("egid")

        col_names = list(df.columns)
        from_idx = len(col_names) // 4 * (self.ksop.quarter - 1)
        to_idx = len(col_names) // 4 * self.ksop.quarter
        filter_cols = col_names[from_idx:to_idx]
        df = df.select(filter_cols)

        X = df.to_numpy().T  # Transpose to get scenarios as rows

        # C. Run Strategy (Numpy -> Indices)
        indices = self.reducer.get_representative_indices(X, self.ksop.n_scenarios)

        log.info(f"Selected scenario indices: {indices}")
        # D. Reconstruction (Indices -> Polars)
        # We select the rows matching the indices
        self.desired_cols = ["egid"] + [col_names[i] for i in indices]

        self.reduced_load_df = load_df.select(self.desired_cols)
        self.reduced_pv_df = pv_df.select(self.desired_cols)

        return self

    def map2scens(
        self,
        egid_id_mapping_file: Path,
        net: pp.pandapowerNet,
        cosφ: float,
        s_base: float,
        seed: int = 42,
    ):
        id_node_mapping = net.load
        external_grid = net.ext_grid["bus"].to_list()
        # E. Return Outputs
        egid_id_mapping = pl.read_csv(egid_id_mapping_file).with_columns(
            pl.col("egid").cast(pl.Utf8)
        )
        id_node_mapping["index"] = id_node_mapping.index
        id_node_mapping_pl = pl.from_dataframe(id_node_mapping[["index", "bus"]])

        reduced_load_df = self.reduced_load_df.join(
            egid_id_mapping[["index", "egid"]], on="egid", how="left"
        ).join(id_node_mapping_pl[["index", "bus"]], on="index", how="left")
        reduced_pv_df = self.reduced_pv_df.join(
            egid_id_mapping[["index", "egid"]], on="egid", how="left"
        ).join(id_node_mapping_pl[["index", "bus"]], on="index", how="left")

        for ex_bus in external_grid:
            if ex_bus not in reduced_load_df["bus"].to_list():
                new_row = pl.DataFrame(
                    {
                        "egid": [f"ext_grid_{ex_bus}"],
                        **{col: [0.0] for col in self.desired_cols[1:]},
                        "index": [-1],
                        "bus": [ex_bus],
                    }
                ).with_columns(pl.col("bus").cast(pl.UInt32))
                reduced_load_df = pl.concat([reduced_load_df, new_row], how="vertical")

        random.seed(seed)
        rng = np.random.default_rng(seed)

        scenarios: Dict[int, pt.DataFrame[LoadData]] = {}
        for ω in range(len(self.desired_cols) - 1):
            q_factor = (1 - cosφ**2) ** 0.5
            df1 = reduced_load_df.select(
                [
                    pl.col("bus").alias("node_id"),
                    (pl.col(self.desired_cols[ω + 1]) / s_base).alias("p_cons_pu"),
                    (pl.col(self.desired_cols[ω + 1]) * q_factor / s_base).alias(
                        "q_cons_pu"
                    ),
                ]
            )
            df2 = reduced_pv_df.select(
                [
                    pl.col("bus").alias("node_id"),
                    (pl.col(self.desired_cols[ω + 1]) / s_base).alias("p_prod_pu"),
                    (pl.col(self.desired_cols[ω + 1]) * q_factor / s_base).alias(
                        "q_prod_pu"
                    ),
                ]
            )
            df = df1.join(df2, on="node_id", how="left")
            random_voltage = (
                rng.uniform(low=0, high=1, size=df.height)
                * (self.ksop.v_bounds[1] - self.ksop.v_bounds[0])
                + self.ksop.v_bounds[0]
                + 1.0
            )
            df = df.select(
                [
                    pl.col("node_id").alias("node_id"),
                    pl.col("p_cons_pu").fill_null(0).alias("p_cons_pu") * 1e3,
                    pl.col("q_cons_pu").fill_null(0).alias("q_cons_pu") * 1e3,
                    pl.col("p_prod_pu").fill_null(0).alias("p_prod_pu") * 1e3,
                    pl.col("q_prod_pu").fill_null(0).alias("q_prod_pu") * 1e3,
                    pl.Series(random_voltage).alias("v_node_sqr_pu"),
                ]
            )

            df = (
                df.sort("node_id")
                .drop_nulls(["node_id"])
                .group_by("node_id")
                .agg(
                    pl.col("p_cons_pu")
                    .sum()
                    .alias("p_cons_pu")
                    .map_elements(lambda x: x if x > 1e-3 else 0.0),
                    pl.col("q_cons_pu")
                    .sum()
                    .alias("q_cons_pu")
                    .map_elements(lambda x: x if x > 1e-3 else 0.0),
                    pl.col("p_prod_pu")
                    .sum()
                    .alias("p_prod_pu")
                    .map_elements(lambda x: x if x > 1e-3 else 0.0),
                    pl.col("q_prod_pu")
                    .sum()
                    .alias("q_prod_pu")
                    .map_elements(lambda x: x if x > 1e-3 else 0.0),
                    pl.col("v_node_sqr_pu").mean().alias("v_node_sqr_pu"),
                )
            ).select(
                [
                    "node_id",
                    "p_cons_pu",
                    "q_cons_pu",
                    "p_prod_pu",
                    "q_prod_pu",
                    "v_node_sqr_pu",
                ]
            )

            df_pt = (
                pt.DataFrame(df)
                .set_model(LoadData)
                .fill_null(strategy="defaults")
                .cast(strict=True)
            )
            df_pt.validate()

            scenarios[ω] = df_pt

        return scenarios


def generate_profile_based_load_scenarios(
    grid: GridCaseModel,
    profiles: ShortTermUncertaintyProfile,
    net: pp.pandapowerNet,
    seed: int,
) -> Dict[int, pt.DataFrame[LoadData]]:
    """
    Generate load scenarios based on predefined profiles using ScenarioPipelineProfile.
    """
    scenario_pipeline = ScenarioPipelineProfile()
    rand_scenarios = scenario_pipeline.process(ksop=profiles).map2scens(
        egid_id_mapping_file=Path(grid.egid_id_mapping_file),
        net=net,
        cosφ=grid.cosφ,
        s_base=grid.s_base,
        seed=seed,
    )
    return rand_scenarios


# --- 4. USAGE EXAMPLE ---
if __name__ == "__main__":
    # --- INPUT DATA ---
    ksop = ShortTermUncertaintyProfile(
        load_profiles=["examples/ieee_33/load_profiles"],
        pv_profile="examples/ieee_33/pv_profiles",
        target_year=2030,
        quarter=4,
        scenario_name=DiscreteScenario.BASIC,
        n_scenarios=10,
    )

    # --- EXECUTION ---

    # 1. Configure the strategy (Nearest to Center)
    strategy = KMeansMedoidReducer()

    # 2. Initialize Pipeline
    pipeline = ScenarioPipelineProfile(strategy)

    net = pp.from_pickle("examples/ieee_33/simple_grid.p")

    # 3. Run Pipeline
    final_scenarios = pipeline.process(ksop=ksop).map2scens(
        egid_id_mapping_file=Path("examples/ieee_33/consumer_egid_idx_mapping.csv"),
        net=net,
        cosφ=0.95,
        s_base=1,
        seed=42,
    )

    log.info(f"Final scenarios generated.")
