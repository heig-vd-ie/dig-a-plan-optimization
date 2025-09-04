# %%
import os

os.chdir(os.getcwd().replace("/src", ""))

from plots import *


my_config = MongoConfig(
    start_collection="run_20250903_114427",
    end_collection="run_20250903_142621",
    mongodb_port=27017,
    mongodb_host="localhost",
    database_name="optimization",
    plot_width=1000,
    plot_height=600,
    histogram_bins=50,
)


client = MyMongoClient(my_config)
client.connect()
client.load_collections()

# %%
objectives_df = client.extract_objectives()

# %%
simulations_df = client.extract_simulations()


# %%
voltage_data = client.extract_voltage_data()
current_data = client.extract_current_data()
real_power_data = client.extract_real_power_data()
reactive_power_data = client.extract_reactive_power_data()
switches_data = client.extract_switches_data()
taps_data = client.extract_taps_data()
r_norm_data = client.extract_r_norm_data()
s_norm_data = client.extract_s_norm_data()

# %%

viz = MyObjectivePlotter(objectives_df, my_config, "objective_value")
fig_hist = viz.create_histogram_plot()
fig_box = viz.create_box_plot()
fig_scatter = viz.create_scatter_plot()
fig_hist.show()
fig_box.show()
fig_scatter.show()
# %%
