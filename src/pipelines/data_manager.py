from data_schema.node_data import NodeData
from data_schema.edge_data import EdgeData
from data_schema import NodeEdgeModel
import patito as pt
import polars as pl
from polars import col as c
from general_function import pl_to_dict, pl_to_dict_with_tuple


class PipelineDataManager:
    """Manages grid data validation and processing"""

    def __init__(self, big_m: float, small_m: float):
        self.__node_data: pt.DataFrame[NodeData] = NodeData.DataFrame(
            schema=NodeData.columns
        ).cast()
        self.__edge_data: pt.DataFrame[EdgeData] = EdgeData.DataFrame(
            schema=EdgeData.columns
        ).cast()
        self.__slack_node: int
        self.big_m: float = big_m
        self.small_m: float = small_m
        self.grid_data_parameters_dict: dict | None = None

    @property
    def node_data(self) -> pt.DataFrame[NodeData]:
        return self.__node_data

    @property
    def edge_data(self) -> pt.DataFrame[EdgeData]:
        return self.__edge_data

    @property
    def slack_node(self) -> int:
        return self.__slack_node

    def add_grid_data(self, grid_data: NodeEdgeModel) -> None:
        """Add and validate grid data"""
        self.__set_node_data(node_data=grid_data.node_data)
        self.__set_edge_data(edge_data=grid_data.edge_data)
        self.__validate_slack_node()
        self.__instantiate_grid_data()

    def __set_node_data(self, node_data: pl.DataFrame):
        """Validate and set node data"""
        old_table: pl.DataFrame = self.__node_data.clear()
        col_list: list[str] = list(
            set(node_data.columns).intersection(set(old_table.columns))
        )
        new_table_pl: pl.DataFrame = pl.concat(
            [old_table, node_data.select(col_list)], how="diagonal_relaxed"
        )
        new_table_pt: pt.DataFrame[NodeData] = (
            pt.DataFrame(new_table_pl)
            .set_model(NodeData)
            .fill_null(strategy="defaults")
            .cast(strict=True)
        )
        new_table_pt.validate()
        self.__node_data = new_table_pt

    def __set_edge_data(self, edge_data: pl.DataFrame):
        """Validate and set edge data"""
        old_table: pl.DataFrame = self.__edge_data.clear()
        col_list: list[str] = list(
            set(edge_data.columns).intersection(set(old_table.columns))
        )
        new_table_pl: pl.DataFrame = pl.concat(
            [old_table, edge_data.select(col_list)], how="diagonal_relaxed"
        )
        new_table_pt: pt.DataFrame[EdgeData] = (
            pt.DataFrame(new_table_pl)
            .set_model(EdgeData)
            .fill_null(strategy="defaults")
            .cast(strict=True)
        )
        new_table_pt.validate()
        self.__edge_data = new_table_pt

    def __validate_slack_node(self):
        """Validate there's exactly one slack node"""
        if self.node_data.filter(c("type") == "slack").height != 1:
            raise ValueError("There must be only one slack node")

        self.__slack_node: int = self.node_data.filter(c("type") == "slack")["node_id"][
            0
        ]

    def __instantiate_grid_data(self):
        """Instantiate the model with the current node and edge data"""
        self.grid_data_parameters_dict = {
            None: {
                "N": {None: self.node_data["node_id"].to_list()},
                "L": {None: self.edge_data["edge_id"].to_list()},
                "S": {
                    None: self.edge_data.filter(c("type") == "switch")[
                        "edge_id"
                    ].to_list()
                },
                "C": {
                    None: self.edge_data.select(
                        pl.concat_list("edge_id", "u_of_edge", "v_of_edge")
                        .map_elements(tuple, return_dtype=pl.Object)
                        .alias("C")
                    )["C"].to_list()
                    + self.edge_data.select(
                        pl.concat_list("edge_id", "v_of_edge", "u_of_edge")
                        .map_elements(tuple, return_dtype=pl.Object)
                        .alias("C")
                    )["C"].to_list()
                },
                "r": pl_to_dict(self.edge_data["edge_id", "r_pu"]),
                "x": pl_to_dict(self.edge_data["edge_id", "x_pu"]),
                "b": pl_to_dict(self.edge_data["edge_id", "b_pu"]),
                "n_transfo": pl_to_dict_with_tuple(
                    pl.concat(
                        [
                            self.edge_data.select(
                                pl.concat_list("edge_id", "u_of_edge", "v_of_edge"),
                                "n_transfo",
                            ),
                            self.edge_data.select(
                                pl.concat_list("edge_id", "v_of_edge", "u_of_edge"),
                                "n_transfo",
                            ),
                        ]
                    )
                ),
                "p_node": pl_to_dict(self.node_data["node_id", "p_node_pu"]),
                "q_node": pl_to_dict(self.node_data["node_id", "q_node_pu"]),
                "i_max": pl_to_dict(self.edge_data["edge_id", "i_max_pu"]),
                "v_min": pl_to_dict(self.node_data["node_id", "v_min_pu"]),
                "v_max": pl_to_dict(self.node_data["node_id", "v_max_pu"]),
                "slack_node": {None: self.slack_node},
                "slack_node_v_sq": {
                    None: 1  # self.node_data.filter(c("type") == "slack")["v_node_sqr_pu"][0]
                },
                "big_m": {None: self.big_m},
                "small_m": {None: self.small_m},
            }
        }
