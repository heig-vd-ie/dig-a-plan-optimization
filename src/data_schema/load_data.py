from typing import Optional
import patito as pt
import polars as pl
from typing_extensions import Literal
from data_schema._constraints import literal_constraint

TYPES = Literal["slack", "pq", "pv"]


class LoadData(pt.Model):
    node_id: int = pt.Field(dtype=pl.Int32, unique=True)
    p_node_pu: float = pt.Field(dtype=pl.Float64, default=0.0)
    q_node_pu: float = pt.Field(dtype=pl.Float64, default=0.0)
    v_node_sqr_pu: Optional[float] = pt.Field(dtype=pl.Float64, default=1.0)
    type: TYPES = pt.Field(
        dtype=pl.Utf8, constraints=literal_constraint(pt.field, TYPES)
    )
