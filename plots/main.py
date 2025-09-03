# %%
import os

os.chdir(os.getcwd().replace("/src", ""))
# %%
from plots import *

config = Config(
    start_collection="run_20250903_114427",
    end_collection="run_20250903_142621",
    mongodb_port=27017,
    mongodb_host="localhost",
    database_name="optimization",
    plot_width=1000,
    plot_height=600,
    histogram_bins=50,
)


my_client = MyMongoClient(config)

# %% Plot objectives
df = my_client.get_dataframe("objectives")

viz = MyObjectivePlotter(df, config, "objectives")

fig_hist = viz.create_histogram_plot()
fig_box = viz.create_box_plot()
fig_scatter = viz.create_scatter_plot()
fig_hist.show()
fig_box.show()
fig_scatter.show()

# %% Plot Investment
df = my_client.get_dataframe("investment_cost")

viz = MyObjectivePlotter(df, config, "investment_cost")

fig_hist = viz.create_histogram_plot()
fig_box = viz.create_box_plot()
fig_scatter = viz.create_scatter_plot()
fig_hist.show()
fig_box.show()
fig_scatter.show()

# %%
