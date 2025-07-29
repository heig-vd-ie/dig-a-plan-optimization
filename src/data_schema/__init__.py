from dataclasses import dataclass, field
import patito as pt
from edge_data import EdgeData
from node_data import NodeData
from load_data import LoadData


@dataclass(frozen=True)
class NodeEdgeModel:
    node_data: pt.DataFrame = field(
        default_factory=lambda: NodeData.DataFrame(schema=NodeData.columns).cast()
    )
    edge_data: pt.DataFrame = field(
        default_factory=lambda: EdgeData.DataFrame(schema=EdgeData.columns).cast()
    )
    load_data: dict[str, pt.DataFrame] = field(default_factory=dict)
