from typing import Optional
import patito as pt
import polars as pl


class LoadData(pt.Model):
    node_id: int = pt.Field(dtype=pl.Int32, unique=True)
    p_node_pu: float = pt.Field(dtype=pl.Float64, default=0.0)
    q_node_pu: float = pt.Field(dtype=pl.Float64, default=0.0)
    v_node_sqr_pu: Optional[float] = pt.Field(dtype=pl.Float64, default=1.0)
