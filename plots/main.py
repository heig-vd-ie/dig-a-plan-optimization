# %%
import os
import pandas as pd

os.chdir(os.getcwd().replace("/src", ""))

from plots import *

if to_extract := False:
    my_config = MongoConfig(
        start_collection="run_20250916_081629",
        end_collection="run_20250916_091726",
        mongodb_port=27017,
        mongodb_host="localhost",
        database_name="optimization",
    )
    client = MyMongoClient(my_config)
    client.connect()
    client.load_collections()
    objectives_df = client.extract_objectives()
    out_of_sample_objectives_df = client.extract_objectives(out_of_sample=True)
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
    out_of_sample_objectives_df.to_parquet(
        ".cache/final_results/out_of_sample_objectives.parquet"
    )
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
    out_of_sample_objectives_df = pd.read_parquet(
        ".cache/final_results/out_of_sample_objectives.parquet"
    )
    simulations_df = pd.read_parquet(".cache/final_results/simulations.parquet")
    voltage_data = pd.read_parquet(".cache/final_results/voltage.parquet")
    current_data = pd.read_parquet(".cache/final_results/current.parquet")
    # real_power_data = pd.read_parquet(".cache/final_results/real_power.parquet")
    # reactive_power_data = pd.read_parquet(".cache/final_results/reactive_power.parquet")
    # switches_data = pd.read_parquet(".cache/final_results/switches.parquet")
    # taps_data = pd.read_parquet(".cache/final_results/taps.parquet")
    r_norm_data = pd.read_parquet(".cache/final_results/r_norm.parquet")
    s_norm_data = pd.read_parquet(".cache/final_results/s_norm.parquet")

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
out_of_sample_objectives_df["objective_value"] = (
    out_of_sample_objectives_df["objective_value"] / 1e3
)  # Convert to M$
fig_hist_oo = viz.create_histogram_plot(
    out_of_sample_objectives_df,
    field="objective_value",
    field_name="Out-of-sample CAPEX (M$)",
    save_name="out_of_sample_objective_histogram",
    nbins=50,
)
fig_hist_oo.show()
fig_box_oo = viz.create_box_plot(
    out_of_sample_objectives_df,
    field="objective_value",
    field_name="Out-of-sample CAPEX (M$)",
    save_name="out_of_sample_objective_boxplot",
)
fig_box_oo.show()

# %%
simulations_df["final_Capacity"] = simulations_df["cap"].apply(
    lambda x: sum([i["out"] for i in x])
)
simulations_df["investment_cost"] = simulations_df["investment_cost"] / 1e3

risk_labels = ["Expectation (α=0.1)", "WorstCase (α=0.1)", "Wasserstein (α=0.1)"]

for kase in ["final_Capacity", "investment_cost"]:
    separate_figs = viz.create_parallel_coordinates_plot(
        simulations_df,
        risk_labels=risk_labels,
        value_col=kase,
        field_name=(
            kase.replace("final_", "").replace("_", " ").title() + "(kW)"
            if kase == "final_Capacity"
            else kase.replace("final_", "").replace("_", " ").title() + " (M$)"
        ),
        stage_col="stage",
        save_name=kase + "_Evolution",
        title_prefix="",
    )

    # Show each plot
    for risk_label, fig in separate_figs.items():
        print(f"Showing plot for: {risk_label}")
        fig.show()
# %%
fig_box = viz.create_box_plot(
    voltage_data,
    field="v_pu",
    with_respect_to="iteration",
    field_name="Voltage (per unit)",
    save_name="voltage_boxplot",
)
fig_box.show()

# %%

fig_box = viz.create_box_plot(
    current_data,
    field="i_pct",
    with_respect_to="stage",
    field_name="Current (per unit)",
    save_name="current_boxplot",
)
fig_box.show()

# %%
