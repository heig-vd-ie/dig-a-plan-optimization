import polars as pl
import pandapower as pp
from general_function import duckdb_to_dict
from power_profiles.download_from_swisstopo import download_estavayer_power_profiles
from config import settings
from enum import Enum


class LoadType(Enum):
    PV = "PV"
    EV = "EV"
    HP = "HP"
    DHW = "DHW"
    TOTAL = "TOTAL"


class ScenarioFactory:

    def __init__(self, kace: str):
        download_estavayer_power_profiles(kace=kace, force=False)
        self.kace = kace

    def _read_parquet(self, file_path: str) -> pl.DataFrame:
        return pl.read_parquet(
            f"{settings.cases[self.kace].load_allocation_folder}/{file_path}.parquet"
        )

    def get_scenario(self):
        dfs = {k.value: self._read_parquet(k.value) for k in LoadType}
        load_duckdb = duckdb_to_dict(settings.cases[self.kace].load_duckdb_file)
        net = pp.from_pickle(settings.cases[self.kace].pandapower_file)
        return (
            dfs,
            load_duckdb,
            net,
        )


if __name__ == "__main__":
    df, load_duckdb, net = ScenarioFactory(kace="estavayer_centre_ville").get_scenario()
    print(df)
    print(load_duckdb)
