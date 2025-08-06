from typing import Optional
import patito as pt
import polars as pl
from typing_extensions import Literal
from data_schema._constraints import literal_constraint

TYPES = Literal["branch", "transformer", "switch"]


class EdgeData(pt.Model):
    eq_fk: str = pt.Field(dtype=pl.Utf8, unique=True)
    edge_id: int = pt.Field(dtype=pl.Int32, unique=True)
    i_base: Optional[float] = pt.Field(dtype=pl.Float64)
    u_of_edge: int = pt.Field(dtype=pl.Int32)
    v_of_edge: int = pt.Field(dtype=pl.Int32)
    r_pu: float = pt.Field(dtype=pl.Float64, default=0.0)
    x_pu: float = pt.Field(dtype=pl.Float64, default=0.0)
    b_pu: float = pt.Field(dtype=pl.Float64, default=0.0)
    g_pu: float = pt.Field(dtype=pl.Float64, default=0.0)
    i_max_pu: Optional[float] = pt.Field(dtype=pl.Float64, default=1.0)
    p_max_pu: Optional[float] = pt.Field(dtype=pl.Float64, default=10.0)
    normal_open: bool = pt.Field(dtype=pl.Boolean, default=False)
    type: TYPES = pt.Field(
        dtype=pl.Utf8, constraints=literal_constraint(pt.field, TYPES)
    )
