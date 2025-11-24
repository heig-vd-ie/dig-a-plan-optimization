import polars as pl
import numpy as np

from helper_functions.general_functions import (
    generate_log,
    modify_string,
)


# Global variable
log = generate_log(name=__name__)


def cast_boolean(col: pl.Expr) -> pl.Expr:
    """
    Cast a column to boolean based on predefined replacements.

    Args:
        col (pl.Expr): The column to cast.

    Returns:
        pl.Expr: The casted boolean column.
    """
    format_str = {
        "1": True,
        "true": True,
        "oui": True,
        "1.0": True,
        "0": False,
        "0.0": False,
        "false": False,
        "vrai": True,
        "non": False,
        "off": False,
        "on": True,
    }
    return (
        col.cast(pl.Utf8)
        .str.to_lowercase()
        .replace_strict(format_str, default=False)
        .cast(pl.Boolean)
    )


def modify_string_col(string_col: pl.Expr, format_str: dict) -> pl.Expr:
    """
    Modify string columns based on a given format dictionary.

    Args:
        string_col (pl.Expr): The string column to modify.
        format_str (dict): The format dictionary containing the string modifications.

    Returns:
        pl.Expr: The modified string column.
    """
    return string_col.map_elements(
        lambda x: modify_string(string=x, format_str=format_str),
        return_dtype=pl.Utf8,
        skip_nulls=True,
    )


def get_transfo_impedance(
    rated_v: pl.Expr, rated_s: pl.Expr, voltage_ratio: pl.Expr
) -> pl.Expr:
    """
    Get the transformer impedance (or resistance if real part) based on the short-circuit tests.

    Args:
        rated_v (pl.Expr): The rated voltage column indicates which side of the transformer the parameters are
        associated with (usually lv side).[V].
        rated_s (pl.Expr): The rated power column [VA].
        voltage_ratio (pl.Expr): The ratio between the applied input voltage to get rated current when transformer
        secondary is short-circuited and the rated voltage [%].

    Returns:
        pl.Expr: The transformer impedance column [Ohm].
    """
    return voltage_ratio / 100 * (rated_v**2) / rated_s


def get_transfo_imaginary_component(module: pl.Expr, real: pl.Expr) -> pl.Expr:
    """
    Get the transformer imaginary component based on the module and real component.

    Args:
        module (pl.Expr): The module column [Ohm or Simens].
        real (pl.Expr): The real component column [Ohm or Simens].

    Returns:
        pl.Expr: The transformer imaginary component column [Ohm or Simens].
    """
    return np.sqrt(module**2 - real**2)
