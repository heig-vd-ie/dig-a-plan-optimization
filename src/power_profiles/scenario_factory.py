import polars as pl
import pandapower as pp
from config import settings
from general_function import duckdb_to_dict
from power_profiles.download_from_swisstopo import download_estavayer_power_profiles
from power_profiles.models import LoadType


class ScenarioFactory:

    def __init__(self, kace: str):
        self.kace = kace
        self.dfs: dict[str, pl.DataFrame] = {}
        self.load_duckdb: dict[str, pl.DataFrame] = {}
        self.net: pp.pandapowerNet | None = None

    def initialize(self):
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


if __name__ == "__main__":
    scenario = ScenarioFactory(kace="estavayer_centre_ville").initialize()
    print(scenario.dfs)
    print(scenario.load_duckdb)
    print(scenario.net)
