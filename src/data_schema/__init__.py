from dataclasses import dataclass
from edge_data import EdgeData
from node_data import NodeData
from load_data import LoadData


@dataclass
class NodeEdgeModel:
    node_data: NodeData
    edge_data: EdgeData
    load_data: dict[str, LoadData]
