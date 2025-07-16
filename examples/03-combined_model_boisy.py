# %% import libraries
import os
from networkx import connected_components
import polars as pl
from data_connector import (
    change_schema_to_dig_a_plan_schema,
    duckdb_to_changes_schema,
)

from pipelines import DigAPlan
from pipelines.configs import CombinedConfig, PipelineType


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
    pl.lit(0.001).alias("r_pu"),
    pl.lit(0.001).alias("x_pu"),
    pl.lit(0).alias("b_pu"),
)

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

# %%
import networkx as nx

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
