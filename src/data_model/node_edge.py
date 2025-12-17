from dataclasses import dataclass, field
from typing import List, Dict, Optional
from typing_extensions import Literal
import polars as pl
import patito as pt

EDGE_TYPES = Literal["branch", "transformer", "switch"]
NODE_TYPES = Literal["slack", "pq", "pv"]


def literal_constraint(field: pl.Expr, values) -> pl.Expr:
    return field.is_in(list(values.__args__)).alias("literal_constraint")


def optional_unique(field: pl.Expr) -> pl.Expr:
    return field.drop_nulls().is_duplicated().sum() == 0


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
    length_km: Optional[float] = pt.Field(dtype=pl.Float64)
    i_max_pu: Optional[float] = pt.Field(dtype=pl.Float64, default=1.0)
    p_max_pu: Optional[float] = pt.Field(dtype=pl.Float64, default=10.0)
    normal_open: bool = pt.Field(dtype=pl.Boolean, default=False)
    taps: list[int] = pt.Field(dtype=pl.List(pl.Int32), default=[100])
    coords: List[float] = pt.Field(dtype=pl.List(pl.Float64))
    x_coords: List[float] = pt.Field(dtype=pl.List(pl.Float64))
    y_coords: List[float] = pt.Field(dtype=pl.List(pl.Float64))
    type: EDGE_TYPES = pt.Field(
        dtype=pl.Utf8, constraints=literal_constraint(pt.field, EDGE_TYPES)
    )


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
    coords: List[float] = pt.Field(dtype=pl.List(pl.Float64))
    type: NODE_TYPES = pt.Field(
        dtype=pl.Utf8, constraints=literal_constraint(pt.field, NODE_TYPES)
    )


class LoadData(pt.Model):
    node_id: int = pt.Field(dtype=pl.Int32, unique=True)
    p_cons_pu: float = pt.Field(dtype=pl.Float64, default=0.0)
    q_cons_pu: float = pt.Field(dtype=pl.Float64, default=0.0)
    p_prod_pu: float = pt.Field(dtype=pl.Float64, default=0.0)
    q_prod_pu: float = pt.Field(dtype=pl.Float64, default=0.0)
    v_node_sqr_pu: Optional[float] = pt.Field(dtype=pl.Float64, default=1.0)


@dataclass
class NodeEdgeModel:
    node_data: pt.DataFrame[NodeData] = field(
        default_factory=lambda: NodeData.DataFrame(schema=NodeData.columns).cast()
    )
    edge_data: pt.DataFrame[EdgeData] = field(
        default_factory=lambda: EdgeData.DataFrame(schema=EdgeData.columns).cast()
    )
    load_data: Dict[int, pt.DataFrame[LoadData]] = field(default_factory=dict)
