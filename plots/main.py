# %%
import os
import pandas as pd

os.chdir(os.getcwd().replace("/src", ""))

from plots import *

if to_extract := False:
    my_config = MongoConfig(
        start_collection="run_20250903_114427",
        end_collection="run_20250903_142621",
        mongodb_port=27017,
        mongodb_host="localhost",
        database_name="optimization",
    )
    client = MyMongoClient(my_config)
    client.connect()
    client.load_collections()
    objectives_df = client.extract_objectives()
    simulations_df = client.extract_simulations()
    voltage_data = client.extract_voltage_data()
    current_data = client.extract_current_data()
    real_power_data = client.extract_real_power_data()
    reactive_power_data = client.extract_reactive_power_data()
    switches_data = client.extract_switches_data()
    taps_data = client.extract_taps_data()
    r_norm_data = client.extract_r_norm_data()
    s_norm_data = client.extract_s_norm_data()
    os.makedirs(".cache/final_results", exist_ok=True)
    objectives_df.to_parquet(".cache/final_results/objectives.parquet")
    simulations_df.to_parquet(".cache/final_results/simulations.parquet")
    voltage_data.to_parquet(".cache/final_results/voltage.parquet")
    current_data.to_parquet(".cache/final_results/current.parquet")
    real_power_data.to_parquet(".cache/final_results/real_power.parquet")
    reactive_power_data.to_parquet(".cache/final_results/reactive_power.parquet")
    switches_data.to_parquet(".cache/final_results/switches.parquet")
    taps_data.to_parquet(".cache/final_results/taps.parquet")
    r_norm_data.to_parquet(".cache/final_results/r_norm.parquet")
    s_norm_data.to_parquet(".cache/final_results/s_norm.parquet")
else:
    objectives_df = pd.read_parquet(".cache/final_results/objectives.parquet")
    simulations_df = pd.read_parquet(".cache/final_results/simulations.parquet")
    # voltage_data = pd.read_parquet(".cache/final_results/voltage.parquet")
    # current_data = pd.read_parquet(".cache/final_results/current.parquet")
    # real_power_data = pd.read_parquet(".cache/final_results/real_power.parquet")
    # reactive_power_data = pd.read_parquet(".cache/final_results/reactive_power.parquet")
    # switches_data = pd.read_parquet(".cache/final_results/switches.parquet")
    # taps_data = pd.read_parquet(".cache/final_results/taps.parquet")
    # r_norm_data = pd.read_parquet(".cache/final_results/r_norm.parquet")
    # s_norm_data = pd.read_parquet(".cache/final_results/s_norm.parquet")

# %%

viz = MyPlotter()
objectives_df["objective_value"] = (
    objectives_df["objective_value"] / 1e3
)  # Convert to M$
fig_hist = viz.create_histogram_plot(
    objectives_df,
    field="objective_value",
    field_name="CAPEX (M$)",
    save_name="objective_histogram",
    nbins=50,
)
fig_hist.show()
fig_box = viz.create_box_plot(
    objectives_df,
    field="objective_value",
    field_name="CAPEX (M$)",
    save_name="objective_boxplot",
)
fig_box.show()

# %%
simulations_df["final_cap"] = simulations_df["cap"].apply(
    lambda x: sum([i["out"] for i in x])
)

risk_labels = ["Expectation (α=0.1)", "WorstCase (α=0.1)", "Wasserstein (α=0.1)"]
separate_figs = viz.create_parallel_coordinates_plot(
    simulations_df,
    risk_labels=risk_labels,
    value_col="final_cap",
    field_name="Capacity (MW)",
    stage_col="stage",
    save_name="Capacity_Evolution",
    title_prefix="",
)

# Show each plot
for risk_label, fig in separate_figs.items():
    print(f"Showing plot for: {risk_label}")
    fig.show()

# %%
# Simulations data visualization
# if not simulations_df.empty and "objective_value" in simulations_df.columns:
#     sim_viz = MyObjectivePlotter(simulations_df, my_config, "objective_value")
#     sim_hist = sim_viz.create_histogram_plot("CAPEX ($)")
#     sim_box = sim_viz.create_box_plot()
#     sim_scatter = sim_viz.create_scatter_plot()
#     sim_hist.update_layout(title="Simulation Objectives Distribution")
#     sim_box.update_layout(title="Simulation Objectives by Risk Method")
#     sim_scatter.update_layout(title="Simulation Objectives vs Iteration")
#     sim_hist.show()
#     sim_box.show()
#     sim_scatter.show()

# # %%
# # Voltage data visualization
# if not voltage_data.empty:
#     # Voltage magnitude distribution
#     if "v_pu" in voltage_data.columns:
#         voltage_viz = MyObjectivePlotter(voltage_data, my_config, "v_pu")
#         voltage_hist = voltage_viz.create_histogram_plot()
#         voltage_box = voltage_viz.create_box_plot()
#         voltage_scatter = voltage_viz.create_scatter_plot()
#         voltage_hist.update_layout(
#             title="Voltage Magnitude (p.u.) Distribution", xaxis_title="Voltage (p.u.)"
#         )
#         voltage_box.update_layout(
#             title="Voltage Magnitude by Risk Method", yaxis_title="Voltage (p.u.)"
#         )
#         voltage_scatter.update_layout(
#             title="Voltage vs Iteration", yaxis_title="Voltage (p.u.)"
#         )
#         voltage_hist.show()
#         voltage_box.show()
#         voltage_scatter.show()

#     # Voltage squared distribution
#     if "v_sq" in voltage_data.columns:
#         v_sq_viz = MyObjectivePlotter(voltage_data, my_config, "v_sq")
#         v_sq_hist = v_sq_viz.create_histogram_plot()
#         v_sq_box = v_sq_viz.create_box_plot()
#         v_sq_hist.update_layout(
#             title="Voltage Squared Distribution", xaxis_title="Voltage Squared"
#         )
#         v_sq_box.update_layout(
#             title="Voltage Squared by Risk Method", yaxis_title="Voltage Squared"
#         )
#         v_sq_hist.show()
#         v_sq_box.show()

# # %%
# # Current data visualization
# if not current_data.empty:
#     # Current magnitude distribution
#     if "i_pu" in current_data.columns:
#         current_viz = MyObjectivePlotter(current_data, my_config, "i_pu")
#         current_hist = current_viz.create_histogram_plot()
#         current_box = current_viz.create_box_plot()
#         current_scatter = current_viz.create_scatter_plot()
#         current_hist.update_layout(
#             title="Current Magnitude (p.u.) Distribution", xaxis_title="Current (p.u.)"
#         )
#         current_box.update_layout(
#             title="Current Magnitude by Risk Method", yaxis_title="Current (p.u.)"
#         )
#         current_scatter.update_layout(
#             title="Current vs Iteration", yaxis_title="Current (p.u.)"
#         )
#         current_hist.show()
#         current_box.show()
#         current_scatter.show()

#     # Current percentage distribution
#     if "i_pct" in current_data.columns:
#         i_pct_viz = MyObjectivePlotter(current_data, my_config, "i_pct")
#         i_pct_hist = i_pct_viz.create_histogram_plot()
#         i_pct_box = i_pct_viz.create_box_plot()
#         i_pct_hist.update_layout(
#             title="Current Percentage Distribution", xaxis_title="Current (%)"
#         )
#         i_pct_box.update_layout(
#             title="Current Percentage by Risk Method", yaxis_title="Current (%)"
#         )
#         i_pct_hist.show()
#         i_pct_box.show()

# # %%
# # Real power data visualization
# if not real_power_data.empty:
#     # Power flow distribution
#     if "p_flow" in real_power_data.columns:
#         power_viz = MyObjectivePlotter(real_power_data, my_config, "p_flow")
#         power_hist = power_viz.create_histogram_plot()
#         power_box = power_viz.create_box_plot()
#         power_scatter = power_viz.create_scatter_plot()
#         power_hist.update_layout(
#             title="Real Power Flow Distribution", xaxis_title="Power Flow (MW)"
#         )
#         power_box.update_layout(
#             title="Real Power Flow by Risk Method", yaxis_title="Power Flow (MW)"
#         )
#         power_scatter.update_layout(
#             title="Real Power vs Iteration", yaxis_title="Power Flow (MW)"
#         )
#         power_hist.show()
#         power_box.show()
#         power_scatter.show()

#     # Power per unit distribution
#     if "p_pu" in real_power_data.columns:
#         p_pu_viz = MyObjectivePlotter(real_power_data, my_config, "p_pu")
#         p_pu_hist = p_pu_viz.create_histogram_plot()
#         p_pu_box = p_pu_viz.create_box_plot()
#         p_pu_hist.update_layout(
#             title="Real Power (p.u.) Distribution", xaxis_title="Power (p.u.)"
#         )
#         p_pu_box.update_layout(
#             title="Real Power (p.u.) by Risk Method", yaxis_title="Power (p.u.)"
#         )
#         p_pu_hist.show()
#         p_pu_box.show()

# # %%
# # Reactive power data visualization
# if not reactive_power_data.empty:
#     # Reactive power flow distribution
#     if "q_flow" in reactive_power_data.columns:
#         reactive_viz = MyObjectivePlotter(reactive_power_data, my_config, "q_flow")
#         reactive_hist = reactive_viz.create_histogram_plot()
#         reactive_box = reactive_viz.create_box_plot()
#         reactive_scatter = reactive_viz.create_scatter_plot()
#         reactive_hist.update_layout(
#             title="Reactive Power Flow Distribution",
#             xaxis_title="Reactive Power (MVAr)",
#         )
#         reactive_box.update_layout(
#             title="Reactive Power by Risk Method", yaxis_title="Reactive Power (MVAr)"
#         )
#         reactive_scatter.update_layout(
#             title="Reactive Power vs Iteration", yaxis_title="Reactive Power (MVAr)"
#         )
#         reactive_hist.show()
#         reactive_box.show()
#         reactive_scatter.show()

# # %%
# # Switches data visualization
# if not switches_data.empty and "value" in switches_data.columns:
#     switches_viz = MyObjectivePlotter(switches_data, my_config, "value")
#     switches_hist = switches_viz.create_histogram_plot()
#     switches_box = switches_viz.create_box_plot()
#     switches_hist.update_layout(
#         title="Switches State Distribution", xaxis_title="Switch State"
#     )
#     switches_box.update_layout(
#         title="Switches by Risk Method", yaxis_title="Switch State"
#     )
#     switches_hist.show()
#     switches_box.show()

# # %%
# # Taps data visualization
# if not taps_data.empty and "value" in taps_data.columns:
#     taps_viz = MyObjectivePlotter(taps_data, my_config, "value")
#     taps_hist = taps_viz.create_histogram_plot()
#     taps_box = taps_viz.create_box_plot()
#     taps_hist.update_layout(
#         title="Transformer Taps Distribution", xaxis_title="Tap Position"
#     )
#     taps_box.update_layout(title="Taps by Risk Method", yaxis_title="Tap Position")
#     taps_hist.show()
#     taps_box.show()

# # %%
# # R-norm convergence visualization
# if not r_norm_data.empty and "value" in r_norm_data.columns:
#     r_norm_viz = MyObjectivePlotter(r_norm_data, my_config, "value")
#     r_norm_hist = r_norm_viz.create_histogram_plot()
#     r_norm_box = r_norm_viz.create_box_plot()
#     r_norm_scatter = r_norm_viz.create_scatter_plot()
#     r_norm_hist.update_layout(
#         title="R-Norm Convergence Distribution", xaxis_title="R-Norm"
#     )
#     r_norm_box.update_layout(title="R-Norm by Risk Method", yaxis_title="R-Norm")
#     r_norm_scatter.update_layout(title="R-Norm vs Iteration", yaxis_title="R-Norm")
#     r_norm_hist.show()
#     r_norm_box.show()
#     r_norm_scatter.show()

# # %%
# # S-norm convergence visualization
# if not s_norm_data.empty and "value" in s_norm_data.columns:
#     s_norm_viz = MyObjectivePlotter(s_norm_data, my_config, "value")
#     s_norm_hist = s_norm_viz.create_histogram_plot()
#     s_norm_box = s_norm_viz.create_box_plot()
#     s_norm_scatter = s_norm_viz.create_scatter_plot()
#     s_norm_hist.update_layout(
#         title="S-Norm Convergence Distribution", xaxis_title="S-Norm"
#     )
#     s_norm_box.update_layout(title="S-Norm by Risk Method", yaxis_title="S-Norm")
#     s_norm_scatter.update_layout(title="S-Norm vs Iteration", yaxis_title="S-Norm")
#     s_norm_hist.show()
#     s_norm_box.show()
#     s_norm_scatter.show()

# # %%
# # Multi-variable comparison plots
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots

# # Voltage vs Current scatter plot
# if not voltage_data.empty and not current_data.empty:
#     if "v_pu" in voltage_data.columns and "i_pu" in current_data.columns:
#         fig_volt_curr = make_subplots(
#             rows=1,
#             cols=2,
#             subplot_titles=("Voltage Distribution", "Current Distribution"),
#             specs=[[{"secondary_y": False}, {"secondary_y": False}]],
#         )

#         # Add voltage histogram
#         fig_volt_curr.add_trace(
#             go.Histogram(x=voltage_data["v_pu"], name="Voltage (p.u.)", nbinsx=30),
#             row=1,
#             col=1,
#         )

#         # Add current histogram
#         fig_volt_curr.add_trace(
#             go.Histogram(x=current_data["i_pu"], name="Current (p.u.)", nbinsx=30),
#             row=1,
#             col=2,
#         )

#         fig_volt_curr.update_layout(
#             title="Voltage and Current Distributions Comparison",
#             showlegend=True,
#             height=500,
#             width=1000,
#         )
#         fig_volt_curr.show()

# # %%
# # Power comparison plot
# if not real_power_data.empty and not reactive_power_data.empty:
#     if "p_flow" in real_power_data.columns and "q_flow" in reactive_power_data.columns:
#         fig_power_comp = make_subplots(
#             rows=1,
#             cols=2,
#             subplot_titles=("Real Power Distribution", "Reactive Power Distribution"),
#             specs=[[{"secondary_y": False}, {"secondary_y": False}]],
#         )

#         # Add real power histogram
#         fig_power_comp.add_trace(
#             go.Histogram(
#                 x=real_power_data["p_flow"], name="Real Power (MW)", nbinsx=30
#             ),
#             row=1,
#             col=1,
#         )

#         # Add reactive power histogram
#         fig_power_comp.add_trace(
#             go.Histogram(
#                 x=reactive_power_data["q_flow"], name="Reactive Power (MVAr)", nbinsx=30
#             ),
#             row=1,
#             col=2,
#         )

#         fig_power_comp.update_layout(
#             title="Real vs Reactive Power Distributions",
#             showlegend=True,
#             height=500,
#             width=1000,
#         )
#         fig_power_comp.show()

# # %%
# # Convergence analysis plot
# if not r_norm_data.empty and not s_norm_data.empty:
#     if "value" in r_norm_data.columns and "value" in s_norm_data.columns:
#         fig_convergence = make_subplots(
#             rows=2,
#             cols=1,
#             subplot_titles=("R-Norm Convergence", "S-Norm Convergence"),
#             specs=[[{"secondary_y": False}], [{"secondary_y": False}]],
#         )

#         # Add R-norm trace
#         if "iteration" in r_norm_data.columns:
#             fig_convergence.add_trace(
#                 go.Scatter(
#                     x=r_norm_data["iteration"],
#                     y=r_norm_data["value"],
#                     mode="markers",
#                     name="R-Norm",
#                     marker=dict(size=6, opacity=0.7),
#                 ),
#                 row=1,
#                 col=1,
#             )

#         # Add S-norm trace
#         if "iteration" in s_norm_data.columns:
#             fig_convergence.add_trace(
#                 go.Scatter(
#                     x=s_norm_data["iteration"],
#                     y=s_norm_data["value"],
#                     mode="markers",
#                     name="S-Norm",
#                     marker=dict(size=6, opacity=0.7),
#                 ),
#                 row=2,
#                 col=1,
#             )

#         fig_convergence.update_layout(
#             title="ADMM Convergence Analysis", showlegend=True, height=800, width=1000
#         )
#         fig_convergence.update_xaxes(title_text="Iteration", row=2, col=1)
#         fig_convergence.update_yaxes(title_text="R-Norm", row=1, col=1)
#         fig_convergence.update_yaxes(title_text="S-Norm", row=2, col=1)
#         fig_convergence.show()

# # %%
