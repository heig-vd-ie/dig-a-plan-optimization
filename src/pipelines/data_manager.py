from data_schema import NodeData, EdgeData, LoadData
from data_schema import NodeEdgeModel
import patito as pt
import polars as pl
from polars import col as c
from general_function import pl_to_dict, pl_to_dict_with_tuple


class PipelineDataManager:
    """Manages grid data validation and processing"""

    def __init__(
        self,
        big_m: float,
        small_m: float,
        ρ: float,
        weight_infeasibility: float = 1.0,
        weight_penalty: float = 1e-6,
        weight_admm_penalty: float = 1.0,
        z: float = 0.0,
        λ: float = 0.0,
    ):

        self.big_m: float = big_m
        self.small_m: float = small_m
        self.ρ = ρ
        self.weight_infeasibility: float = weight_infeasibility
        self.weight_admm_penalty: float = weight_admm_penalty
        self.weight_penalty: float = weight_penalty
        self.z: float = z
        self.λ: float = λ

        # patito tables for static schemas
        self.__node_data: pt.DataFrame[NodeData] = NodeData.DataFrame(
            schema=NodeData.columns
        ).cast()
        self.__edge_data: pt.DataFrame[EdgeData] = EdgeData.DataFrame(
            schema=EdgeData.columns
        ).cast()
        self.__slack_node: int

        # will hold the scenario‑indexed loads
        self.__load_data: dict[str, pt.DataFrame[LoadData]] = {}
        # final data dict for Pyomo
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
        # 1) static node + edge
        self.__set_node_data(node_data=grid_data.node_data)
        self.__set_edge_data(edge_data=grid_data.edge_data)
        self.__validate_slack_node()

        # 2) scenario loads
        self.__set_load_data(grid_data.load_data)

        # 3) now build the full parameters dict
        self.__instantiate_grid_data()

    def __set_node_data(self, node_data: pt.DataFrame[NodeData]):
        """Validate and set static node data"""
        old_table: pl.DataFrame = self.__node_data.clear()
        col_list: list[str] = list(
            set(node_data.as_polars().columns).intersection(set(old_table.columns))
        )
        new_table_pl: pl.DataFrame = pl.concat(
            [old_table, node_data.as_polars().select(col_list)], how="diagonal_relaxed"
        )
        new_table_pt: pt.DataFrame[NodeData] = (
            pt.DataFrame(new_table_pl)
            .set_model(NodeData)
            .fill_null(strategy="defaults")
            .cast(strict=True)
        )
        new_table_pt.validate()
        self.__node_data = new_table_pt

    def __set_edge_data(self, edge_data: pt.DataFrame[EdgeData]):
        """Validate and set edge data"""
        old_table: pl.DataFrame = self.__edge_data.clear()
        col_list: list[str] = list(
            set(edge_data.as_polars().columns).intersection(set(old_table.columns))
        )
        new_table_pl: pl.DataFrame = pl.concat(
            [old_table, edge_data.as_polars().select(col_list)], how="diagonal_relaxed"
        )
        new_table_pt: pt.DataFrame[EdgeData] = (
            pt.DataFrame(new_table_pl)
            .set_model(EdgeData)
            .fill_null(strategy="defaults")
            .cast(strict=True)
        )
        new_table_pt.validate()
        self.__edge_data = new_table_pt

    def __set_load_data(self, load_data: dict[str, pt.DataFrame[LoadData]]):
        """
        Validate each scenario DataFrame against data_schema.load_data.NodeData
        and store the validated Polars frames in self.__load_data.
        """
        validated: dict[str, pt.DataFrame[LoadData]] = {}
        for scen_id, data_F in load_data.items():
            df_pt = (
                pt.DataFrame(data_F.as_polars())
                .set_model(LoadData)
                .fill_null(strategy="defaults")
                .cast(strict=True)
            )
            df_pt.validate()
            # as_polars() is equivalent to .to_polars() in recent patito
            validated[scen_id] = df_pt

        self.__load_data = validated

    def __validate_slack_node(self):
        """Validate there's exactly one slack node"""
        if self.node_data.filter(c("type") == "slack").height != 1:
            raise ValueError("There must be only one slack node")

        self.__slack_node: int = self.node_data.filter(c("type") == "slack")["node_id"][
            0
        ]

    def __instantiate_grid_data(self):
        """
        Build a Pyomo data dict that includes:
          - SCEN set
          - static sets N, L, S, nS, C
          - static parameters (r,x,b,n_transfo,i_max,v_min,v_max,...)
          - scenario parameters p_node[s,n], q_node[s,n]
          - ADMM params del_param[s,l], u_param[s,l]
        """
        # --- sets & keys ---
        scen_ids = list(self.__load_data.keys())  # type: ignore
        node_ids = self.node_data["node_id"].to_list()
        edge_ids = self.edge_data["edge_id"].to_list()
        switch_ids = self.edge_data.filter(c("type") == "switch")["edge_id"].to_list()
        # non_switch = [e for e in edge_ids if e not in switch_ids]

        # build directed arcs list C
        c_fwd = self.edge_data.select(
            pl.concat_list("edge_id", "u_of_edge", "v_of_edge")
            .map_elements(tuple, return_dtype=pl.Object)
            .alias("C")
        )["C"].to_list()
        c_rev = self.edge_data.select(
            pl.concat_list("edge_id", "v_of_edge", "u_of_edge")
            .map_elements(tuple, return_dtype=pl.Object)
            .alias("C")
        )["C"].to_list()
        c_tuples = c_fwd + c_rev

        self.grid_data_parameters_dict = {
            scen_id: {
                None: {
                    # sets
                    "N": {None: node_ids},
                    "L": {None: edge_ids},
                    "S": {None: switch_ids},
                    "C": {None: c_tuples},
                    # static line params
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
                    "i_max": pl_to_dict(self.edge_data["edge_id", "i_max_pu"]),
                    # static node params
                    "v_min": pl_to_dict(self.node_data["node_id", "v_min_pu"]),
                    "v_max": pl_to_dict(self.node_data["node_id", "v_max_pu"]),
                    # slack‑bus
                    "slack_node": {None: self.__slack_node},
                    "slack_node_v_sq": {
                        None: float(
                            self.__load_data[scen_id].filter(
                                c("node_id") == self.__slack_node
                            )["v_node_sqr_pu"][0]
                        )
                    },
                    # scenario loads
                    "p_node": pl_to_dict(
                        self.__load_data[scen_id]["node_id", "p_node_pu"]
                    ),
                    "q_node": pl_to_dict(
                        self.__load_data[scen_id]["node_id", "q_node_pu"]
                    ),
                    # ADMM & big‑M parameters
                    "big_m": {None: self.big_m},
                    "small_m": {None: self.small_m},
                    "weight_infeasibility": {None: self.weight_infeasibility},
                    "weight_admm_penalty": {None: self.weight_admm_penalty},
                    "weight_penalty": {None: self.weight_penalty},
                    "ρ": {None: self.ρ},
                    "z": {None: self.z},
                    "λ": {None: self.λ},
                }
            }
            for scen_id in scen_ids
        }
