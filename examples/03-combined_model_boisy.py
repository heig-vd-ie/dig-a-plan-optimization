# %% import libraries
import os
import pandapower as pp
from polars import col as c
import polars as pl
from local_data_exporter import (
    change_schema_to_dig_a_plan_schema,
    duckdb_to_changes_schema,
)
from local_data_exporter import pandapower_to_dig_a_plan_schema

from pipelines import DigAPlan
from pipelines.configs import CombinedConfig, PipelineType


# ensure working directory is project root
os.chdir(os.getcwd().replace("/src", ""))
os.environ["GRB_LICENSE_FILE"] = os.environ["HOME"] + "/gurobi_license/gurobi.lic"

# %% set parameters
if USE_SIMPLIFIED_GRID := True:
    net = pp.from_pickle(".cache/boisy_grid_simplified.p")
    base_grid_data = pandapower_to_dig_a_plan_schema(net)
else:
    change_schema = duckdb_to_changes_schema(".cache/boisy_grid.db")
    base_grid_data = change_schema_to_dig_a_plan_schema(change_schema, 1000)

# %% convert pandapower grid to DigAPlan grid data
base_grid_data.node_data = base_grid_data.node_data.with_columns(
    pl.lit(0.01).alias("p_node_pu"),
    pl.lit(0.01).alias("q_node_pu"),
)
base_grid_data.edge_data = base_grid_data.edge_data.with_columns(
    pl.lit(0.001).alias("r_pu"),
    pl.lit(0.001).alias("x_pu"),
    pl.lit(0).alias("b_pu"),
)
base_grid_data.node_data = base_grid_data.node_data.with_columns(
    # (c("p_node_pu")*5e-1).alias("p_node_pu"),
    (c("p_node_pu") * 1e-1).alias("q_node_pu")
)

base_grid_data.edge_data = base_grid_data.edge_data.with_columns(
    pl.when(c(col) < 1e-3).then(pl.lit(0)).otherwise(c(col)).alias(col)
    for col in ["b_pu", "r_pu", "x_pu"]
).with_columns(
    pl.lit(1.0).alias("n_transfo"),
    c("normal_open").fill_null(False),
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
