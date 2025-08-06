from dataclasses import dataclass, field
from typing import List, Dict
import patito as pt
from data_schema.edge_data import EdgeData
from data_schema.node_data import NodeData
from data_schema.load_data import LoadData


@dataclass
class NodeEdgeModel:
    node_data: pt.DataFrame[NodeData] = field(
        default_factory=lambda: NodeData.DataFrame(schema=NodeData.columns).cast()
    )
    edge_data: pt.DataFrame[EdgeData] = field(
        default_factory=lambda: EdgeData.DataFrame(schema=EdgeData.columns).cast()
    )
    load_data: Dict[int, pt.DataFrame[LoadData]] = field(default_factory=dict)
    taps: List[int] = field(default_factory=list)
