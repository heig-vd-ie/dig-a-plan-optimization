from typing import Optional
import patito as pt
import polars as pl


class NodeData(pt.Model):
    cn_fk: str = pt.Field(dtype=pl.Utf8, unique=True)
    node_id: int = pt.Field(dtype=pl.Int32, unique=True)
    i_base: Optional[float] = pt.Field(dtype=pl.Float64, default=None)
    s_base: Optional[float] = pt.Field(dtype=pl.Float64, default=None)
    v_base: float = pt.Field(dtype=pl.Float64)
    v_min_pu: float = pt.Field(dtype=pl.Float64, default=0.9)
    v_max_pu: float = pt.Field(dtype=pl.Float64, default=1.1)
    p_node_max_pu: float = pt.Field(dtype=pl.Float64, default=0.0)
    p_node_min_pu: float = pt.Field(dtype=pl.Float64, default=0.0)
    q_node_max_pu: float = pt.Field(dtype=pl.Float64, default=0.0)
    q_node_min_pu: float = pt.Field(dtype=pl.Float64, default=0.0)
