import polars as pl
from polars import col as c
import pandapower as pp
from helpers.general import pl_to_dict
from pathlib import Path
from data_model import ShortTermUncertaintyProfile
from data_model.kace import GridCaseModel


class ScenarioPotentials:
    def __init__(
        self,
        profiles: ShortTermUncertaintyProfile,
        net: pp.pandapowerNet,
        egid_id_mapping: pl.DataFrame,
    ):
        self.profiles = profiles
        self.net = net
        self.egid_id_mapping = egid_id_mapping.select("index", "egid").with_columns(
            c("egid").cast(pl.Utf8)
        )

    def compute_potential_load(self) -> dict[int, float]:
        loads_df = []
        for load_profile_path in self.profiles.load_profiles:
            df = self.compute_potential(load_profile_path)
            loads_df.append(df)
        df_cap: pl.DataFrame = pl.concat(loads_df)
        return self.map_to_potential_dict(df_cap)

    def compute_potential_pv(self) -> dict[int, float]:
        df_cap = self.compute_potential(self.profiles.pv_profile)
        return self.map_to_potential_dict(df_cap)

    def map_to_potential_dict(self, df: pl.DataFrame) -> dict[int, float]:
        self.net.load["index"] = self.net.load.index
        df_temp = df.join(self.egid_id_mapping, on="egid", how="full").select(
            "egid", "capacity", "index"
        )
        all_nodes = self.net.bus.index.to_list()
        df_temp = (
            df_temp.join(
                pl.from_pandas(self.net.load),
                on="index",
                how="full",
            )
            .drop_nulls("bus")
            .select("bus", "capacity")
            .group_by("bus")
            .sum()
        )
        cap_dict = pl_to_dict(df_temp.select("bus", "capacity"))
        cap_dict_full = {node: cap_dict.get(node, 0.0) for node in all_nodes}
        return cap_dict_full

    def compute_potential(self, profile_path: str):
        profile_path_typed = Path(profile_path)
        profile_files = [p for p in profile_path_typed.iterdir() if p.is_file()]
        profile_files_filtered = [
            p
            for p in profile_files
            if int(str(p).split("_")[-1].replace(".parquet", ""))
            >= self.profiles.target_year
        ]
        df: None | pl.DataFrame = None
        for p in profile_files_filtered:
            df_temp = (
                pl.read_parquet(p)
                .select(
                    c("egid"),
                    pl.mean_horizontal(pl.all().exclude("egid")).alias("minimum"),
                )
                .with_columns(c("minimum").alias("maximum"))
            )
            if df is None:
                df = df_temp
            df = df.join(df_temp, on="egid", how="full", suffix="_temp").select(
                c("egid"),
                pl.min_horizontal("minimum", "minimum_temp").alias("minimum"),
                pl.max_horizontal("maximum", "maximum_temp").alias("maximum"),
            )
        if df is None:
            raise ValueError("There is no profile with respect to the target year")
        return df.select(c("egid"), (c("maximum") - c("minimum")).alias("capacity"))


def generate_scenario_potentials(
    grid: GridCaseModel,
    profiles: ShortTermUncertaintyProfile | None = None,
) -> tuple[dict[int, float], dict[int, float]]:
    """
    Generate dummy potentials for load and PV based on node IDs.

    This is a placeholder function. Replace with actual logic to compute potentials.
    """
    net = pp.from_pickle(grid.pp_file)
    egid_id_mapping = pl.read_csv(grid.egid_id_mapping_file)

    if profiles is None:
        load_potential = {node: 5.0 for node in net.bus.index.to_list()}
        pv_potential = {node: 1.0 for node in net.bus.index.to_list()}
    else:
        scenario_profile = ScenarioPotentials(profiles, net, egid_id_mapping)
        load_potential = scenario_profile.compute_potential_load()
        pv_potential = scenario_profile.compute_potential_pv()

    return load_potential, pv_potential


if __name__ == "__main__":
    from helpers import generate_log

    log = generate_log(__name__)
    grid = GridCaseModel()
    load_potential, pv_potential = generate_scenario_potentials(
        grid=grid, profiles=ShortTermUncertaintyProfile()
    )
    log.info("Scenario potentials generated successfully.")
