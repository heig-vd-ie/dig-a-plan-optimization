# %% import libraries
import os
import networkx as nx
from polars import col as c
from networkx import connected_components
import polars as pl
from local_data_exporter import (
    change_schema_to_dig_a_plan_schema,
    duckdb_to_changes_schema,
)

from pipelines import DigAPlan
from pipelines.configs import CombinedConfig, PipelineType
from networkx_function import generate_nx_edge, get_connected_edges_data


# ensure working directory is project root
os.chdir(os.getcwd().replace("/src", ""))
os.environ["GRB_LICENSE_FILE"] = os.environ["HOME"] + "/gurobi_license/gurobi.lic"

# %% set parameters
change_schema = duckdb_to_changes_schema(".cache/boisy_grid.db")

base_grid_data = change_schema_to_dig_a_plan_schema(change_schema, 1000)
base_grid_data.node_data = base_grid_data.node_data.with_columns(
    pl.lit(0.01).alias("p_node_pu"),
    pl.lit(0.01).alias("q_node_pu"),
)
base_grid_data.edge_data = base_grid_data.edge_data.with_columns(
    # pl.lit(0.001).alias("r_pu"),
    # pl.lit(0.001).alias("x_pu"),
    pl.lit(0).alias("b_pu"),
)

# %%
nx_graph = nx.Graph()
_ = base_grid_data.edge_data.with_columns(
    pl.struct(pl.all()).pipe(generate_nx_edge, nx_graph=nx_graph)
)

max_n_trafo = 1
edge_id = (
    get_connected_edges_data(nx_graph=nx_graph)
    .filter(c("graph_id") != 0)["edge_id"]
    .to_list()
)
node_id = (
    get_connected_edges_data(nx_graph=nx_graph)
    .filter(c("graph_id") != 0)
    .unpivot(on=["u_of_edge", "v_of_edge"])["value"]
    .to_list()
)

base_grid_data.edge_data = base_grid_data.edge_data.filter(~c("edge_id").is_in(edge_id))
base_grid_data.node_data = base_grid_data.node_data.filter(
    ~c("node_id").is_in(node_id + [252])
)

base_grid_data.edge_data = base_grid_data.edge_data.with_columns(
    pl.when(c(col) < 1e-3).then(pl.lit(0)).otherwise(c(col)).alias(col)
    for col in ["b_pu", "r_pu", "x_pu"]
).with_columns(
    pl.lit(1.0).alias("n_transfo"),
    c("normal_open").fill_null(False).alias("normal_open"),
)

base_grid_data.node_data = base_grid_data.node_data.with_columns(
    # (c("p_node_pu")*5e-1).alias("p_node_pu"),
    (c("p_node_pu") * 1e-1).alias("q_node_pu")
)
edge_id = (
    base_grid_data.edge_data.unpivot(
        on=["u_of_edge", "v_of_edge"], index=["edge_id", "type"]
    )
    .filter(c("value").is_unique())
    .filter(c("type") == "switch")["edge_id"]
    .to_list()
)

base_grid_data.edge_data = base_grid_data.edge_data.with_columns(
    pl.when(c("edge_id").is_in(edge_id))
    .then(pl.lit(False))
    .otherwise(c("normal_open"))
    .alias("normal_open")
)

base_grid_data.edge_data = base_grid_data.edge_data.filter(~c("normal_open"))

# %%

g = nx.Graph()
base_grid_data.node_data.to_pandas().apply(
    lambda row: g.add_node(row["node_id"]), axis=1
)
base_grid_data.edge_data.filter(pl.col("type") == "branch").to_pandas().apply(
    lambda row: g.add_edge(row["u_of_edge"], row["v_of_edge"]), axis=1
)
base_grid_data.edge_data.filter(
    (pl.col("type") == "switch") & (~pl.col("normal_open"))
).to_pandas().apply(lambda row: g.add_edge(row["u_of_edge"], row["v_of_edge"]), axis=1)

nx.is_connected(g)  # Check if the graph is connected
connected_component = list(nx.connected_components(g))


# %% initialize DigAPlan

config = CombinedConfig(
    verbose=True,
    big_m=1000,
    small_m=0.1,
    pipeline_type=PipelineType.COMBINED,
)
dig_a_plan = DigAPlan(config=config)

# %% add grid data and solve the combined model
dig_a_plan.add_grid_data(base_grid_data)
dig_a_plan.solve_model()  # one‐shot solve

# %% extract and compare results
# Switch status
switches = dig_a_plan.result_manager.extract_switch_status()
# Node voltages
voltages = dig_a_plan.result_manager.extract_node_voltage()
# Line currents
currents = dig_a_plan.result_manager.extract_edge_current()
active_power_flow = dig_a_plan.result_manager.extract_edge_active_power_flow()
reactive_power_flow = dig_a_plan.result_manager.extract_edge_reactive_power_flow()
