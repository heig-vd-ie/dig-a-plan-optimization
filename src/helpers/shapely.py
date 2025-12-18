import os
import json
from typing import Optional
from shapely import Geometry

from shapely.ops import transform
from shapely.geometry import shape


from pyproj import CRS, Transformer


def shape_coordinate_transformer(
    shape: Geometry, srid_from: int, srid_to: int
) -> Geometry:
    """
    Transform the coordinates of geometries from one CRS to another.

    Args:
        shape (Geometry): The Polars expression containing geometries.
        srid_from (int): The source spatial reference system identifier.
        srid_to (int): The target spatial reference system identifier.

    Returns:
        pl.Expr: A Polars expression with transformed geometries.
    """
    transformer = Transformer.from_crs(
        crs_from=CRS(f"EPSG:{srid_from}"), crs_to=CRS(f"EPSG:{srid_to}"), always_xy=True
    ).transform
    return transform(transformer, shape)


def load_shape_from_geo_json(
    file_name: str, srid_from: Optional[str] = None, srid_to: Optional[str] = None
) -> Geometry:
    """
    Load a shape from a GeoJSON file and optionally transform its coordinates.

    Args:
        file_name (str): The path to the GeoJSON file.
        srid_from (Optional[str], optional): The source spatial reference system identifier. Defaults to None.
        srid_to (Optional[str], optional): The target spatial reference system identifier. Defaults to None.

    Returns:
        Geometry: The loaded shape.

    Raises:
        FileNotFoundError: If the GeoJSON file is not found.
        ValueError: If only one of srid_from or srid_to is provided.
    """
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"File {file_name} not found")
    with open(file_name) as f:
        loading_shape = json.load(f)
    geo_shape: Geometry = shape(loading_shape["features"][0]["geometry"])

    if (srid_from is None) | (srid_to is None):
        return geo_shape
    elif (srid_from is not None) and (srid_to is not None):
        return shape_coordinate_transformer(geo_shape, srid_from=srid_from, srid_to=srid_to)  # type: ignore
    else:
        raise ValueError("Both srid_from and srid_to must be provided or None.")
