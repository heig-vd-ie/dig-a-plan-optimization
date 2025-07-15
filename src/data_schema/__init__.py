from dataclasses import dataclass
import polars as pl


@dataclass
class NodeEdgeModel:
    node_data: pl.DataFrame
    edge_data: pl.DataFrame
