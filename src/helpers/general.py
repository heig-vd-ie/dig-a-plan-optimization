"""
Auxiliary functions
"""

import logging
import os
import uuid
import coloredlogs
import polars as pl
import logging
from polars import col as c
import re
import tqdm
import duckdb
import pandas as pd
import geopandas as gpd
from shapely import from_wkt

NAMESPACE_UUID: uuid.UUID = uuid.UUID("{bc4d4e0c-98c9-11ec-b909-0242ac120002}")
SWISS_SRID: int = 2056


def generate_log(name: str, log_level: str = "info") -> logging.Logger:
    """
    Generate a logger with the specified name and log level.

    Args:
        name (str): The name of the logger.
        log_level (str, optional): The log level. Defaults to "info".

    Returns:
        logging.Logger: The generated logger.
    """
    log = logging.getLogger(name)
    coloredlogs.install(level=log_level)
    return log


def build_non_existing_dirs(file_path: str):
    """
    Build non-existing directories for a given file path.

    Args:
        file_path (str): The file path.

    Returns:
        bool: True if directories were created successfully.
    """
    file_path = os.path.normpath(file_path)
    # Split the path into individual directories
    dirs = file_path.split(os.sep)
    # Check if each directory exists and create it if it doesn't
    current_path = ""
    for directory in dirs:
        current_path = os.path.join(current_path, directory)
        if not os.path.exists(current_path):
            os.mkdir(current_path)
    return True


def pl_to_dict(df: pl.DataFrame) -> dict:
    """
    Convert a Polars DataFrame with two columns into a dictionary. It is assumed that the
    first column contains the keys and the second column contains the values. The keys must
    be unique but Null values will be filtered.

    Args:
        df (pl.DataFrame): Polars DataFrame with two columns.

    Returns:
        dict: Dictionary representation of the DataFrame.

    Raises:
        ValueError: If the DataFrame does not have exactly two columns or if the keys are not unique.
    """

    if df.shape[1] != 2:
        raise ValueError("DataFrame is not composed of two columns")

    columns_name = df.columns[0]
    df = df.drop_nulls(columns_name)
    if df[columns_name].is_duplicated().sum() != 0:
        raise ValueError("Key values are not unique")
    return dict(df.rows())


def pl_to_dict_with_tuple(df: pl.DataFrame) -> dict:
    """
    Convert a Polars DataFrame with two columns into a dictionary where the first column
    contains tuples as keys and the second column contains the values.

    Args:
        df (pl.DataFrame): Polars DataFrame with two columns.

    Returns:
        dict: Dictionary representation of the DataFrame with tuples as keys.

    Raises:
        ValueError: If the DataFrame does not have exactly two columns.

    Example:
    >>> import polars as pl
    >>> data = {'key': [[1, 2], [3, 4], [5, 6]], 'value': [10, 20, 30]}
    >>> df = pl.DataFrame(data)
    >>> pl_to_dict_with_tuple(df)
    {(1, 2): 10, (3, 4): 20, (5, 6): 30}
    """
    if df.shape[1] != 2:
        raise ValueError("DataFrame is not composed of two columns")
    return dict(map(lambda data: (tuple(data[0]), data[1]), df.rows()))


def table_to_gpkg(
    table: pl.DataFrame, gpkg_file_name: str, layer_name: str, srid: int = SWISS_SRID
):
    """
    Save a Polars DataFrame as a GeoPackage file. As GeoPackage does not support list columns,
    the list columns are joined into a single string separated with a comma.

    Args:
        table (pl.DataFrame): The Polars DataFrame.
        gpkg_file_name (str): The GeoPackage file name.
        layer_name (str): The layer name.
        srid (int, optional): The SRID. Defaults to SWISS_SRID.
    """
    list_columns: list[str] = [
        name
        for name, col_type in dict(table.schema).items()
        if type(col_type) == pl.List
    ]
    table_pd: pd.DataFrame = table.with_columns(
        c(list_columns).cast(pl.List(pl.Utf8)).list.join(", ")
    ).to_pandas()

    table_pd["geometry"] = table_pd["geometry"].apply(from_wkt)
    table_pd = table_pd[table_pd.geometry.notnull()]
    table_gpd: gpd.GeoDataFrame = gpd.GeoDataFrame(
        table_pd.dropna(axis=0, subset="geometry"), crs=srid
    )  # type: ignore
    table_gpd = table_gpd[~table_gpd["geometry"].is_empty]  # type: ignore
    # Save gpkg without logging
    logger = logging.getLogger("pyogrio")
    previous_level = logger.level
    logger.setLevel(logging.WARNING)
    table_gpd.to_file(gpkg_file_name, layer=layer_name)
    logger.setLevel(previous_level)


def dict_to_gpkg(data: dict, file_path: str, srid: int = SWISS_SRID):
    """
    Save a dictionary of Polars DataFrames as a GeoPackage file.

    Args:
        data (dict): The dictionary of Polars DataFrames.
        file_path (str): The GeoPackage file path.
        srid (int, optional): The SRID. Defaults to SWISS_SRID.
    """
    with tqdm.tqdm(range(1), ncols=100, desc="Save input data in gpkg format") as pbar:
        for layer_name, table in data.items():
            if isinstance(table, pl.DataFrame):
                if not table.is_empty():
                    table_to_gpkg(
                        table=table,
                        gpkg_file_name=file_path,
                        layer_name=layer_name,
                        srid=srid,
                    )
        pbar.update()


def dict_to_duckdb(data: dict[str, pl.DataFrame], file_path: str):
    """
    Save a dictionary of Polars DataFrames as a DuckDB file.

    Args:
        data (dict[str, pl.DataFrame]): The dictionary of Polars DataFrames.
        file_path (str): The DuckDB file path.
    """
    build_non_existing_dirs(os.path.dirname(file_path))
    if os.path.exists(file_path):
        os.remove(file_path)
    with duckdb.connect(file_path) as con:
        con.execute("SET TimeZone='UTC'")
        pbar = tqdm.tqdm(
            data.items(), ncols=150, desc="Save dictionary into duckdb file"
        )
        for table_name, table_pl in pbar:
            query = f"CREATE TABLE {table_name} AS SELECT * FROM table_pl"
            con.execute(query)


def duckdb_to_dict(file_path: str) -> dict:
    """
    Load a DuckDB file into a dictionary of Polars DataFrames.

    Args:
        file_path (str): The DuckDB file path.

    Returns:
        dict: The dictionary of Polars DataFrames.
    """
    schema_dict: dict[str, pl.DataFrame] = {}  # type: ignore

    with duckdb.connect(database=file_path) as con:
        con.execute("SET TimeZone='UTC'")
        query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        pbar = tqdm.tqdm(
            con.execute(query).fetchall(),
            ncols=150,
            desc="Read and validate tables from {} file".format(
                os.path.basename(file_path)
            ),
        )
        for table_name in pbar:
            query: str = f"SELECT * FROM {table_name[0]}"
            schema_dict[table_name[0]] = con.execute(query).pl()

    return schema_dict


def modify_string(string: str, format_str: dict) -> str:
    """
    Modify a string by replacing substrings according to a format dictionary
    -   Input could contains RegEx.
    -   The replacement is done in the order of the dictionary keys.

    Args:
        string (str): Input string.
        format_str (dict): Dictionary containing the substrings to be replaced and their replacements.

    Returns:
        str: Modified string.
    """

    for str_in, str_out in format_str.items():
        string = re.sub(str_in, str_out, string)
    return string


def safe_first(lst: list, default_val: float = 0.0) -> float:
    """
    Safely get the first element of a list. If the list is empty, return None.

    Args:
        lst (list): Input list.
    Returns:
        The first element of the list or default_val if the list is empty.
    """
    if len(lst) > 0:
        return lst[0]
    return default_val
