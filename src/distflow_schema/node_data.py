from typing import Optional
import patito as pt
import polars as pl



class NodeData(pt.Model):
    uuid: str = pt.Field(dtype=pl.Utf8, unique=True)
    node_id: int = pt.Field(dtype=pl.Int32, unique=True)
    v_base: int = pt.Field(dtype=pl.Float64)
    i_base: int = pt.Field(dtype=pl.Float64)
    p_node_pu: float = pt.Field(dtype=pl.Float64, default=0.0)
    q_node_pu: float = pt.Field(dtype=pl.Float64, default=0.0)
    p_node_max_pu: float = pt.Field(dtype=pl.Float64, default=0.0)
    p_node_min_pu: float = pt.Field(dtype=pl.Float64, default=0.0)
    q_node_max_pu: float = pt.Field(dtype=pl.Float64, default=0.0)
    q_node_min_pu: float = pt.Field(dtype=pl.Float64, default=0.0)
    v_node_sqr_pu: Optional[float] = pt.Field(dtype=pl.Float64)