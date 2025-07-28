# %% --- imports ---
import os
import polars as pl
import pandapower as pp
import patito as pt
from dataclasses import is_dataclass

from polars import col as c

# ---- your modules (adjust paths/imports to your repo layout) ----
from local_data_exporter import pandapower_to_dig_a_plan_schema
from data_schema import NodeEdgeModel
from data_schema.node_data import NodeData as NodeStatic
from data_schema.load_data import NodeData as NodeLoad


# %% --- config ---
PP_NET_PATH = "data/simple_grid.p"
LOAD_FACTOR = 1.0
TEST_CONFIG = [
    {"line_list": [], "switch_list": []},
    {"line_list": [6, 9], "switch_list": [25, 28]},
    {"line_list": [2, 6, 9], "switch_list": [21, 25, 28]},
    {"line_list": [16], "switch_list": [35]},
    {"line_list": [1], "switch_list": [20]},
    {"line_list": [10], "switch_list": [29]},
    {"line_list": [7, 11], "switch_list": [26, 30]},
]
NB_TEST = 0
EXPECTED_SCENARIOS = None
ZERO_SLACK = True


# %% --- config --


def validate_node_edge_model(
    model: NodeEdgeModel,
    expected_n_scenarios: int | None = None,
    zero_slack: bool = True,
):
    assert is_dataclass(model), "Returned object is not a dataclass (NodeEdgeModel?)"
    assert hasattr(model, "node_data")
    assert hasattr(model, "edge_data")
    assert hasattr(model, "load_data")

    node_df: pl.DataFrame = model.node_data
    edge_df: pl.DataFrame = model.edge_data
    scenarios: dict[str, pl.DataFrame] = model.load_data

    # ---- sizes
    assert node_df.height > 0, "node_data is empty"
    assert edge_df.height > 0, "edge_data is empty"
    if expected_n_scenarios is not None:
        assert (
            len(scenarios) == expected_n_scenarios
        ), f"Expected {expected_n_scenarios} scenarios, got {len(scenarios)}"

    # ---- schema validation for static nodes
    pt.DataFrame(node_df).set_model(NodeStatic).cast(strict=True).validate()

    # ---- single slack check
    slack_nodes = node_df.filter(pl.col("type") == "slack")
    assert (
        slack_nodes.height == 1
    ), f"Expected exactly 1 slack node, found {slack_nodes.height}"
    slack_node_id = int(slack_nodes["node_id"][0])

    # ---- edges connect existing nodes
    nid_set = set(node_df["node_id"].to_list())
    missing_u = edge_df.filter(~pl.col("u_of_edge").is_in(nid_set))
    missing_v = edge_df.filter(~pl.col("v_of_edge").is_in(nid_set))
    assert (
        missing_u.height == 0 and missing_v.height == 0
    ), "Some edges reference unknown node_ids"

    # ---- validate scenarios
    for sid, sdf in scenarios.items():
        # schema
        pt.DataFrame(sdf).set_model(NodeLoad).cast(strict=True).validate()

        # bounds
        assert (
            sdf["p_node_pu"] >= sdf["p_node_min_pu"]
        ).all(), f"p lower bound violated in scenario {sid}"
        assert (
            sdf["p_node_pu"] <= sdf["p_node_max_pu"]
        ).all(), f"p upper bound violated in scenario {sid}"
        assert (
            sdf["q_node_pu"] >= sdf["q_node_min_pu"]
        ).all(), f"q lower bound violated in scenario {sid}"
        assert (
            sdf["q_node_pu"] <= sdf["q_node_max_pu"]
        ).all(), f"q upper bound violated in scenario {sid}"

        # zero slack if enforced
        if zero_slack:
            slack_rows = sdf.filter(pl.col("node_id") == slack_node_id)
            if slack_rows.height > 0:
                assert (
                    float(slack_rows["p_node_pu"][0]) == 0.0
                ), f"Slack p != 0 in scenario {sid}"
                assert (
                    float(slack_rows["q_node_pu"][0]) == 0.0
                ), f"Slack q != 0 in scenario {sid}"

    print("✔ NodeEdgeModel validated: shapes, schema, bounds, connectivity, slack.")


def small_summary(model: NodeEdgeModel, n_first_scenarios: int = 3):
    node_df: pl.DataFrame = model.node_data
    edge_df: pl.DataFrame = model.edge_data
    scenarios: dict[str, pl.DataFrame] = model.load_data

    print("\n--- SUMMARY ---")
    print(f"Nodes:     {node_df.height}")
    print(f"Edges:     {edge_df.height}")
    print(f"Scenarios: {len(scenarios)}")
    print("First 5 nodes:")
    print(node_df.head())
    print("First 5 edges:")
    print(edge_df.head())
    # show first scenario head
    if scenarios:
        # sort scenario ids numerically when possible
        sorted_sids = sorted(
            scenarios.keys(), key=lambda x: int(x) if x.isdigit() else x
        )
        print("\nScenario IDs:", sorted_sids)

        for sid in sorted_sids[:n_first_scenarios]:
            print(f"\nFirst 5 rows of scenario {sid}:")
            print(scenarios[sid].head())
    print("-------------\n")


# %% --- main ---
if __name__ == "__main__":

    os.chdir(os.getcwd().replace("/src", ""))

    # load pandapower net
    net = pp.from_pickle(PP_NET_PATH)

    # apply your test config
    net["load"]["p_mw"] = net["load"]["p_mw"] * LOAD_FACTOR
    net["load"]["q_mvar"] = net["load"]["q_mvar"] * LOAD_FACTOR

    net["line"].loc[:, "max_i_ka"] = 1.0
    net["line"].loc[TEST_CONFIG[NB_TEST]["line_list"], "max_i_ka"] = 1e-2

    # run conversion
    base_grid_data = pandapower_to_dig_a_plan_schema(net)

    # validate
    validate_node_edge_model(
        model=base_grid_data,
        expected_n_scenarios=EXPECTED_SCENARIOS,
        zero_slack=ZERO_SLACK,
    )

    small_summary(base_grid_data)

    print("All tests passed ✅")
