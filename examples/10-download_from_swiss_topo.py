# %%
from typing import Optional
from shapely import Geometry
import os
from swiss_topo_api import get_all_layers_from_polygon
from general_function import dict_to_gpkg, dict_to_duckdb
from shapely_function import load_shape_from_geo_json
from swiss_topo_utility.constant import SWISS_SRID, GPS_SRID
from config import settings

os.chdir(os.getcwd().replace("/src", ""))

# %%

name: str = "estavayer"
layer_list: Optional[list[str]] = ["regbl"]

shape_file_path: str = settings.cases["estavayer_centre_ville"].geojson_file
input_polygon: Geometry = load_shape_from_geo_json(
    file_name=shape_file_path, srid_from=GPS_SRID, srid_to=SWISS_SRID
)

downloaded_data = get_all_layers_from_polygon(
    polygon=input_polygon, layer_list=layer_list
)
dict_to_gpkg(data=downloaded_data, file_path="qgis/downloaded_data.gpkg")
dict_to_duckdb(data=downloaded_data, file_path=f".cache/duckdb/{name}.db")
