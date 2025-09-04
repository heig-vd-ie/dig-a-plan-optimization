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

objectives_df = client.extract_objectives()

# %%

viz = MyObjectivePlotter(objectives_df, my_config, "objective_value")
fig_hist = viz.create_histogram_plot()
fig_box = viz.create_box_plot()
fig_scatter = viz.create_scatter_plot()
fig_hist.show()
fig_box.show()
fig_scatter.show()
# %%
