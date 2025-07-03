# %%
import os
import pandapower as pp
import polars as pl
from polars import col as c

from general_function import dict_to_duckdb
from pipelines.dig_a_plan_model_test import DigAPlanTest
from pipelines.dig_a_plan_complete_master import DigAPlan

from data_display.grid_plotting import plot_grid_from_pandapower
from data_display.output_processing import compare_dig_a_plan_with_pandapower
from data_connector import change_schema_to_dig_a_plan_schema, duckdb_to_changes_schema
from twindigrid_changes.schema import ChangesSchema
from networkx_function import generate_nx_edge, get_connected_edges_data, generate_bfs_tree_with_edge_data, get_all_edge_data
import networkx as nx

from pyomo_utility import extract_optimization_results

import matplotlib.pyplot as plt

os.chdir(os.getcwd().replace("/src", ""))
os.environ['GRB_LICENSE_FILE'] = os.environ["HOME"] + "/gurobi_license/gurobi.lic"
# os.environ['MOSEKLM_LICENSE_FILE'] = os.environ["HOME"] + "/mosek/mosek.lic"

# %%
from matplotlib.pylab import normal


change_schema: ChangesSchema = duckdb_to_changes_schema(file_path=".cache/input_data/boisy_grid.db")
# 2) build the *original* schema once

base_grid_data = change_schema_to_dig_a_plan_schema(change_schema = change_schema, s_base=1e6)

nx_graph = nx.Graph()
_ = base_grid_data["edge_data"].with_columns(
    pl.struct(pl.all()).pipe(generate_nx_edge, nx_graph=nx_graph)
)

max_n_trafo = 1
edge_id  = get_connected_edges_data(nx_graph = nx_graph).filter(c("graph_id")!= 0)["edge_id"].to_list()
node_id = get_connected_edges_data(nx_graph = nx_graph).filter(c("graph_id")!= 0)\
    .unpivot(on=["u_of_edge", "v_of_edge"])["value"].to_list()

base_grid_data["edge_data"] = base_grid_data["edge_data"].filter(~c("edge_id").is_in(edge_id))
base_grid_data["node_data"] = base_grid_data["node_data"].filter(~c("node_id").is_in(node_id + [252]))

base_grid_data["edge_data"] = base_grid_data["edge_data"].with_columns(
        pl.when(c(col) < 1e-3).then(pl.lit(0)).otherwise(c(col)).alias(col) for col in ["b_pu", "r_pu", "x_pu"]
    ).with_columns(
        pl.lit(1.0).alias("n_transfo"),
        c("normal_open").fill_null(False).alias("normal_open"),
    )
    
base_grid_data["node_data"] = base_grid_data["node_data"].with_columns(
    # (c("p_node_pu")*5e-1).alias("p_node_pu"),
    (c("p_node_pu")*1e-1).alias("q_node_pu")
    )


# %%
edge_id = base_grid_data["edge_data"]\
    .unpivot(on=["u_of_edge", "v_of_edge"], index=["edge_id", "type"])\
    .filter(c("value").is_unique())\
    .filter(c("type")== "switch")["edge_id"].to_list()
    
base_grid_data["edge_data"] = base_grid_data["edge_data"].with_columns(
    pl.when(c("edge_id").is_in(edge_id)).then(pl.lit(False)).otherwise(c("normal_open")).alias("normal_open")
)

base_grid_data["edge_data"] = base_grid_data["edge_data"].filter(~c("normal_open"))

# %%
dict_to_duckdb(base_grid_data, file_path=".cache/input_data/start_boisy_grid.db")

# %%
nx_graph = nx.Graph()
_ = base_grid_data["edge_data"].filter(~c("normal_open")).select(
    pl.struct("edge_id", "u_of_edge", "v_of_edge").pipe(generate_nx_edge, nx_graph=nx_graph)
)
print(nx.is_connected(nx_graph))
print(nx.is_tree(nx_graph))

# %%
