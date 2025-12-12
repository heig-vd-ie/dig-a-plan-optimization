from typing import Optional
import patito as pt
import polars as pl
from typing_extensions import Literal
from data_schema._constraints import literal_constraint

TYPES = Literal["slack", "pq", "pv"]


class NodeData(pt.Model):
    cn_fk: str = pt.Field(dtype=pl.Utf8, unique=True)
    node_id: int = pt.Field(dtype=pl.Int32, unique=True)
    i_base: Optional[float] = pt.Field(dtype=pl.Float64, default=None)
    s_base: Optional[float] = pt.Field(dtype=pl.Float64, default=None)
    v_base: float = pt.Field(dtype=pl.Float64)
    min_vm_pu: float = pt.Field(dtype=pl.Float64, default=0.9)
    max_vm_pu: float = pt.Field(dtype=pl.Float64, default=1.1)
    cons_installed: float = pt.Field(dtype=pl.Float64, default=1.0)
    prod_installed: float = pt.Field(dtype=pl.Float64, default=1.0)

    type: TYPES = pt.Field(
        dtype=pl.Utf8, constraints=literal_constraint(pt.field, TYPES)
    )
