import polars as pl
import pandapower as pp
from config import settings
from data_exporter.duckdb_to_change_schema import duckdb_to_changes_schema
from power_profiles.model import LoadProfileType


def get_scenario(kase: str):
    pv_df = pl.read_parquet(f"{settings.cases[kase].load_allocation_folder}/PV.parquet")
    changes_schema = duckdb_to_changes_schema(settings.cases[kase].geojson_file)
    net = pp.from_pickle(settings.cases[kase].pandapower_file)
    return pv_df, changes_schema, net


if __name__ == "__main__":
    df, changes_schema, net = get_scenario("estavayer_centre_ville")
    print(df)
    print(changes_schema)
