from data_model import NodeData, EdgeData, LoadData, NodeEdgeModel4Reconfiguration
import patito as pt
import polars as pl
from polars import col as c
from helper_functions import pl_to_dict


class PipelineDataManager:
    """Manages grid data validation and processing"""

    def __init__(
        self,
        big_m: float,
        ε: float,
        ρ: float,
        voll: float = 1.0,
        volp: float = 1.0,
        γ_infeasibility: float = 1.0,
        γ_admm_penalty: float = 1.0,
        γ_trafo_loss: float = 1.0,
        z: dict[int, float] | None = None,
        λ: dict[int, float] | None = None,
        all_scenarios: bool = False,
    ):

        self.big_m: float = big_m
        self.ε: float = ε
        self.ρ = ρ
        self.γ_infeasibility: float = γ_infeasibility
        self.γ_admm_penalty: float = γ_admm_penalty
        self.γ_trafo_loss: float = γ_trafo_loss
        self.z: dict[int, float] | None = z
        self.λ: dict[int, float] | None = λ
        self.voll: float = voll
        self.volp: float = volp
        self.all_scenarios: bool = all_scenarios

        # patito tables for static schemas
        self.__node_data: pt.DataFrame[NodeData] = NodeData.DataFrame(
            schema=NodeData.columns
        ).cast()
        self.__edge_data: pt.DataFrame[EdgeData] = EdgeData.DataFrame(
            schema=EdgeData.columns
        ).cast()
        self.__slack_node: int

        # will hold the scenario‑indexed loads
        self.load_data: dict[int, pt.DataFrame[LoadData]] = {}
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

    def add_grid_data(self, grid_data: NodeEdgeModel4Reconfiguration) -> None:
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

    def __set_load_data(self, load_data: dict[int, pt.DataFrame[LoadData]]):
        """
        Validate each scenario DataFrame against data_schema.load_data.NodeData
        and store the validated Polars frames in self.load_data.
        """
        validated: dict[int, pt.DataFrame[LoadData]] = {}
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

        self.load_data = validated

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
        """
        # --- sets & keys ---
        scen_ids = list(self.load_data.keys())  # type: ignore
        node_ids = self.node_data["node_id"].to_list()
        edge_ids = self.edge_data["edge_id"].to_list()
        switch_ids = self.edge_data.filter(c("type") == "switch")["edge_id"].to_list()
        transformer_ids = self.edge_data.filter(c("type") == "transformer")[
            "edge_id"
        ].to_list()
        line_ids = self.edge_data.filter(c("type") == "branch")["edge_id"].to_list()

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

        number_of_lines = len([e for e in edge_ids if e not in switch_ids])

        tr_taps = (
            self.edge_data.filter(c("type") == "transformer")
            .select(
                pl.struct("edge_id", "taps")
                .map_elements(
                    lambda x: [(x["edge_id"], y) for y in x["taps"]],
                    return_dtype=pl.Object,
                )
                .alias("taps")
            )["taps"]
            .to_list()
        )
        tr_taps_flat = [item for sublist in tr_taps for item in sublist]

        self.grid_data_parameters_dict = {
            scen_id: {
                None: {
                    # sets
                    "Ω": {None: scen_ids if self.all_scenarios else [scen_id]},
                    "N": {None: node_ids},
                    "E": {None: edge_ids},
                    "L": {None: line_ids},
                    "S": {None: switch_ids},
                    "C": {None: c_tuples},
                    "Tr": {None: transformer_ids},
                    "TrTaps": {None: tr_taps_flat},
                    # static line params
                    "r": pl_to_dict(self.edge_data["edge_id", "r_pu"]),
                    "x": pl_to_dict(self.edge_data["edge_id", "x_pu"]),
                    "b": pl_to_dict(self.edge_data["edge_id", "b_pu"]),
                    "number_of_lines": {None: number_of_lines},
                    "i_max": pl_to_dict(self.edge_data["edge_id", "i_max_pu"]),
                    # static node params
                    "v_min": pl_to_dict(self.node_data["node_id", "min_vm_pu"]),
                    "v_max": pl_to_dict(self.node_data["node_id", "max_vm_pu"]),
                    # slack‑bus
                    "slack_node": {None: [self.__slack_node]},
                    "slack_node_v_sq": {
                        s: self.load_data[s].filter(
                            c("node_id") == self.__slack_node
                        )["v_node_sqr_pu"][0]
                        for s in (scen_ids if self.all_scenarios else [scen_id])
                    },
                    # scenario loads
                    "p_node_cons": {
                        (n, s): self.load_data[s]["node_id", "p_cons_pu"].filter(
                            c("node_id") == n
                        )["p_cons_pu"][0]
                        for n in node_ids
                        for s in (scen_ids if self.all_scenarios else [scen_id])
                    },
                    "q_node_cons": {
                        (n, s): self.load_data[s]["node_id", "q_cons_pu"].filter(
                            c("node_id") == n
                        )["q_cons_pu"][0]
                        for n in node_ids
                        for s in (scen_ids if self.all_scenarios else [scen_id])
                    },
                    "p_node_prod": {
                        (n, s): self.load_data[s]["node_id", "p_prod_pu"].filter(
                            c("node_id") == n
                        )["p_prod_pu"][0]
                        for n in node_ids
                        for s in (scen_ids if self.all_scenarios else [scen_id])
                    },
                    "q_node_prod": {
                        (n, s): self.load_data[s]["node_id", "q_prod_pu"].filter(
                            c("node_id") == n
                        )["q_prod_pu"][0]
                        for n in node_ids
                        for s in (scen_ids if self.all_scenarios else [scen_id])
                    },
                    "node_cons_installed_param": {
                        n: self.node_data.filter(c("node_id") == n)["cons_installed"][0]
                        for n in node_ids
                    },
                    "node_prod_installed_param": {
                        n: self.node_data.filter(c("node_id") == n)["prod_installed"][0]
                        for n in node_ids
                    },
                    # ADMM & big‑M parameters
                    "big_m": {None: self.big_m},
                    "ε": {None: self.ε},
                    "γ_infeasibility": {None: self.γ_infeasibility},
                    "γ_admm_penalty": {None: self.γ_admm_penalty},
                    "γ_trafo_loss": {None: self.γ_trafo_loss},
                    "ρ": {None: self.ρ},
                    "z": (
                        self.z
                        if self.z is not None
                        else {switch_id: 0.0 for switch_id in switch_ids}
                    ),
                    "λ": (
                        self.λ
                        if self.λ is not None
                        else {switch_id: 0.0 for switch_id in switch_ids}
                    ),
                    "voll": {None: self.voll},
                    "volp": {None: self.volp},
                }
            }
            for scen_id in scen_ids
        }
