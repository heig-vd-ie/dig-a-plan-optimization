import os
from typing import Optional
from shapely import Geometry
from swiss_topo_api import get_all_layers_from_polygon
from general_function import dict_to_gpkg, dict_to_duckdb
from shapely_function import load_shape_from_geo_json
from swiss_topo_utility.constant import SWISS_SRID, GPS_SRID
from config import settings


def download_estavayer_power_profiles(kace: str, force: bool) -> None:
    """
    Downloads power profiles for a specified case from the SwissTopo API and saves them in GPKG and DuckDB formats.

    Retrieves electrical power plant data, roof PV potential, and non-industrial heat demand for the region
    defined by the case polygon. The results are stored for further analysis.

    Args:
        kace (str): Case identifier.
        force (bool): If True, overwrite existing files.

    Requires:
        - swiss_topo_api
        - shapely
        - general_function (dict_to_gpkg, dict_to_duckdb)
        - shapely_function (load_shape_from_geo_json)
        - swiss_topo_utility.constant (SWISS_SRID, GPS_SRID)
        - config (settings)
    """

    if os.path.exists(settings.cases[kace].load_gpkg_file) and not force:
        print(
            f"File {settings.cases[kace].load_gpkg_file} already exists. Use force=True to overwrite."
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
    dict_to_gpkg(data=downloaded_data, file_path=settings.cases[kace].load_gpkg_file)
    dict_to_duckdb(
        data=downloaded_data,
        file_path=settings.cases[kace].load_duckdb_file,
    )


if __name__ == "__main__":
    download_estavayer_power_profiles(kace="estavayer_centre_ville", force=True)
