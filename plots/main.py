# %%
from plots import Config, MyMongoClient, MyObjectivePlotter


config = Config(
    start_collection="run_20250902_110438",
    end_collection="run_20250902_130518",
    mongodb_port=27017,
    mongodb_host="localhost",
    database_name="optimization",
    plot_width=1000,
    plot_height=600,
    histogram_bins=50,
)


my_client = MyMongoClient(config)

# %%
df = my_client.get_dataframe("objectives")

viz = MyObjectivePlotter(df, config)

fig_hist = viz.create_histogram_plot()
fig_box = viz.create_box_plot()
fig_line = viz.create_iteration_plot()
fig_scatter = viz.create_scatter_plot()
fig_hist.show()
fig_box.show()
fig_line.show()
fig_scatter.show()
