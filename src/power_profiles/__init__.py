import os
from typing import Optional
from shapely import Geometry
from swiss_topo_api import get_all_layers_from_polygon
from general_function import dict_to_gpkg, dict_to_duckdb
from shapely_function import load_shape_from_geo_json
from swiss_topo_utility.constant import SWISS_SRID, GPS_SRID
from config import settings


def download_estavayer_power_profiles(
    kace: str = "estavayer", force: bool = False
) -> None:
    """
    Download power profiles for Estavayer case from SwissTopo API and save them in GPKG and DuckDB formats.

    This function retrieves electrical power plant data, roof PV potential, and non-industrial heat demand
    for the Estavayer region using a specified polygon shape. The data is then saved in both GPKG and DuckDB
    formats for further analysis.

    Requirements:
        - The `swiss_topo_api` package must be installed.
        - The `shapely` package must be installed.
        - The `general_function` module must be available with `dict_to_gpkg` and `dict_to_duckdb` functions.
        - The `shapely_function` module must be available with `load_shape_from_geo_json` function.
        - The `swiss_topo_utility.constant` module must be available with `SWISS_SRID` and `GPS_SRID` constants.
        - The `config` module must be available with the necessary settings.
    """

    if os.path.exists(settings.outputs[kace].gpkg_file) and not force:
        print(
            f"File {settings.outputs[kace].gpkg_file} already exists. Use force=True to overwrite."
        )
        return None

    layer_list: Optional[list[str]] = [
        "electrical_power_plant",
        "roof_pv_poterntial",
        "non_industrial_heat_demand",
    ]

    shape_file_path: str = settings.cases[kace].geojson_file
    input_polygon: Geometry = load_shape_from_geo_json(
        file_name=shape_file_path, srid_from=GPS_SRID, srid_to=SWISS_SRID
    )

    downloaded_data = get_all_layers_from_polygon(
        polygon=input_polygon, layer_list=layer_list
    )
    dict_to_gpkg(data=downloaded_data, file_path=settings.outputs[kace].gpkg_file)
    dict_to_duckdb(
        data=downloaded_data,
        file_path=settings.outputs[kace].duckdb_file,
    )


if __name__ == "__main__":
    download_estavayer_power_profiles(force=True)
