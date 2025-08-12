# %%
import copy
import os

os.chdir(os.getcwd().replace("/src", ""))
# %%
from examples import *

# %% set parameters

net = pp.from_pickle("data/simple_grid.p")
grid_data = pandapower_to_dig_a_plan_schema(net, number_of_random_scenarios=10)
groups = {
    0: [19, 20, 21, 29, 32, 35],
    1: [35, 30, 33, 25, 26, 27],
    2: [27, 32, 22, 23, 34],
    3: [31, 24, 28, 21, 22, 23],
    4: [34, 26, 25, 24, 31],
}


# %% Configure ADMM pipeline
config = ADMMConfig(
    verbose=False,
    pipeline_type=PipelineType.ADMM,
    solver_name="gurobi",
    solver_non_convex=2,
    big_m=1e3,
    ε=1,
    ρ=2.0,
    γ_infeasibility=1e6,
    γ_admm_penalty=1.0,
    groups=groups,
    max_iters=10,
    μ=10.0,
    τ_incr=2.0,
    τ_decr=2.0,
)

dap = DigAPlanADMM(config=config)
dap.add_grid_data(grid_data)


# %% Run ADMM
dap.model_manager.solve_model()
# %% Consensus switch states (one value per switch)
print(dap.model_manager.zδ_variable)
print(dap.model_manager.zζ_variable)
# %% compare DigAPlan results with pandapower results
node_data, edge_data = compare_dig_a_plan_with_pandapower(
    dig_a_plan=dap, net=net, from_z=True
)
# %% plot the grid annotated with DigAPlan results
fig = plot_grid_from_pandapower(net, dap, from_z=True)

# %% Check results
all_voltages = []
for ω in range(len(dap.model_manager.admm_model_instances)):
    voltages = dap.result_manager.extract_node_voltage(ω)
    voltages = voltages.with_columns(
        pl.lit(ω).alias("scenario"),
    )
    all_voltages.append(voltages)

voltage_df = pl.concat(all_voltages).with_columns(pl.col("node_id").cast(pl.Int64))
# %% Run ADMM with fixed switches
dap2 = copy.deepcopy(dap)

dap2.model_manager.solve_model(fixed_switches=True)
# %% Check results
all_voltages = []
for ω in range(len(dap2.model_manager.admm_model_instances)):
    voltages = dap2.result_manager.extract_node_voltage(ω)
    voltages = voltages.with_columns(
        pl.lit(ω).alias("scenario"),
    )
    all_voltages.append(voltages)

voltage_df2 = pl.concat(all_voltages).with_columns(pl.col("node_id").cast(pl.Int64))


voltage_df_labeled = voltage_df.with_columns(pl.lit("ADMM").alias("method"))
voltage_df2_labeled = voltage_df2.with_columns(pl.lit("Fixed Switches").alias("method"))

# Combine both datasets
combined_voltage_df = pl.concat([voltage_df_labeled, voltage_df2_labeled])

import plotly.express as px

# Convert to pandas for plotly
combined_df_pd = combined_voltage_df.to_pandas()

# Create interactive box plot
fig = px.box(
    combined_df_pd,
    x="node_id",
    y="v_pu",
    color="method",
    title="Voltage Distribution by Bus: ADMM vs Fixed Switches",
    labels={"node_id": "Bus ID", "v_pu": "Voltage (p.u.)", "method": "Method"},
    hover_data=["scenario"],
)

# Add reference lines for voltage limits
fig.add_hline(
    y=0.95,
    line_dash="dash",
    line_color="red",
    annotation_text="Min Voltage Limit (0.95 p.u.)",
)
fig.add_hline(
    y=1.05,
    line_dash="dash",
    line_color="red",
    annotation_text="Max Voltage Limit (1.05 p.u.)",
)
fig.add_hline(
    y=1.0,
    line_dash="solid",
    line_color="green",
    annotation_text="Nominal Voltage (1.0 p.u.)",
)

# Update layout for better visualization
fig.update_layout(
    width=1200,
    height=600,
    xaxis_title="Bus ID",
    yaxis_title="Voltage (p.u.)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified",
)

# Sort x-axis by bus ID
fig.update_xaxes(categoryorder="category ascending")

fig.show()

# %% Check results
all_currents = []
for ω in range(len(dap.model_manager.admm_model_instances)):
    currents = dap.result_manager.extract_edge_current(ω)
    currents = currents.with_columns(
        pl.lit(ω).alias("scenario"),
    )
    all_currents.append(currents)

current_df = pl.concat(all_currents).with_columns(pl.col("edge_id").cast(pl.Int64))
# %% Check results
all_currents = []
for ω in range(len(dap2.model_manager.admm_model_instances)):
    currents = dap2.result_manager.extract_edge_current(ω)
    currents = currents.with_columns(
        pl.lit(ω).alias("scenario"),
    )
    all_currents.append(currents)

current_df2 = pl.concat(all_currents).with_columns(pl.col("edge_id").cast(pl.Int64))


current_df_labeled = current_df.with_columns(pl.lit("ADMM").alias("method"))
current_df2_labeled = current_df2.with_columns(pl.lit("Fixed Switches").alias("method"))

# Combine both datasets
combined_current_df = pl.concat([current_df_labeled, current_df2_labeled])

import plotly.express as px

# Convert to pandas for plotly
combined_df_pd = combined_current_df.to_pandas()

# Create interactive box plot
fig = px.box(
    combined_df_pd,
    x="edge_id",
    y="i_pu",
    color="method",
    title="Current Distribution by Edge: ADMM vs Fixed Switches",
    labels={"edge_id": "Edge ID", "i_pu": "Current (p.u.)", "method": "Method"},
    hover_data=["scenario"],
)


# Update layout for better visualization
fig.update_layout(
    width=1200,
    height=600,
    xaxis_title="Edge ID",
    yaxis_title="Current (p.u.)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified",
)

# Sort x-axis by bus ID
fig.update_xaxes(categoryorder="category ascending")

fig.show()

# %% Check results
all_currents = []
for ω in range(len(dap.model_manager.admm_model_instances)):
    currents = dap.result_manager.extract_edge_current(ω)
    currents = currents.with_columns(
        pl.lit(ω).alias("scenario"),
    )
    all_currents.append(currents)

current_df = pl.concat(all_currents).with_columns(pl.col("edge_id").cast(pl.Int64))
# %% Check results
all_powers = []
for ω in range(len(dap.model_manager.admm_model_instances)):
    powers = dap.result_manager.extract_edge_reactive_power_flow(ω)
    powers = powers.with_columns(
        pl.lit(ω).alias("scenario"),
    )
    all_powers.append(powers)

power_df = (
    pl.concat(all_powers)
    .with_columns(pl.col("edge_id").cast(pl.Int64))
    .filter(pl.col("from_node_id") < pl.col("to_node_id"))
)
all_powers = []
for ω in range(len(dap2.model_manager.admm_model_instances)):
    powers = dap2.result_manager.extract_edge_reactive_power_flow(ω)
    powers = powers.with_columns(
        pl.lit(ω).alias("scenario"),
    )
    all_powers.append(powers)

power_df2 = (
    pl.concat(all_powers)
    .with_columns(pl.col("edge_id").cast(pl.Int64))
    .filter(pl.col("from_node_id") < pl.col("to_node_id"))
)


power_df_labeled = power_df.with_columns(pl.lit("ADMM").alias("method"))
power_df2_labeled = power_df2.with_columns(pl.lit("Fixed Switches").alias("method"))

# Combine both datasets
combined_power = pl.concat([power_df_labeled, power_df2_labeled])

import plotly.express as px

# Convert to pandas for plotly
combined_df_pd = combined_power.to_pandas()

# Create interactive box plot
fig = px.box(
    combined_df_pd,
    x="edge_id",
    y="q_pu",
    color="method",
    title="Power Distribution by Edge: ADMM vs Fixed Switches",
    labels={"edge_id": "Edge ID", "q_pu": "Power (p.u.)", "method": "Method"},
    hover_data=["scenario"],
)


# Update layout for better visualization
fig.update_layout(
    width=1200,
    height=600,
    xaxis_title="Edge ID",
    yaxis_title="Power (p.u.)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified",
)

# Sort x-axis by bus ID
fig.update_xaxes(categoryorder="category ascending")

fig.show()

# %%
