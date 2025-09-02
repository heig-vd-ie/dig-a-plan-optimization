# %%
import sys
import os

# Ensure we're in the plots directory for local imports
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import Config
from analyzer import OptimizationAnalyzer
import visualizer as viz
import data_processor as dp

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

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

print("Configuration loaded:")
print(f"- Database: {config.mongodb_host}:{config.mongodb_port}/{config.database_name}")
print(f"- Collection range: {config.start_collection} to {config.end_collection}")
print(f"- Plot dimensions: {config.plot_width}x{config.plot_height}")

analyzer = OptimizationAnalyzer(config)

try:
    analyzer.run_full_analysis()
    df_temp = analyzer.get_dataframe()
    print("✅ Analysis completed successfully!")
    print(f"📊 Loaded {len(analyzer.collections)} collections")
    if df_temp is not None:
        print(f"📈 Processed {len(df_temp)} data points")
    else:
        print("📈 No data points processed")
except Exception as e:
    print(f"❌ Error during analysis: {e}")
    print("Please check your MongoDB connection and collection names.")

df = analyzer.get_dataframe()

if df is not None:
    print("Dataset Overview:")
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nRisk Methods: {df['risk_method'].unique()}")
    print(f"\nIteration Range: {df['iteration'].min()} - {df['iteration'].max()}")

    print("\nFirst 5 rows:")
    print(df.head())
else:
    print("No data available. Please check the previous cell for errors.")

if df is not None:
    summary = analyzer.get_summary()
    print("📊 Summary Statistics by Risk Method:")
    print(summary)

    print("\n📈 Overall Objective Statistics:")
    print(df["objective"].describe())

if df is not None:
    fig_hist = viz.create_histogram_plot(df, config)
    fig_hist.show()
else:
    print("No data available for plotting.")

if df is not None:
    fig_box = viz.create_box_plot(df, config)
    fig_box.show()
else:
    print("No data available for plotting.")

if df is not None:
    fig_line = viz.create_iteration_plot(df, config)
    fig_line.show()
else:
    print("No data available for plotting.")

if df is not None:
    fig_scatter = viz.create_scatter_plot(df, config)
    fig_scatter.show()
else:
    print("No data available for plotting.")

if df is not None:
    available_methods = df["risk_method"].unique()
    print(f"Available risk methods: {available_methods}")

if df is not None:
    print(f"Iteration range: {df['iteration'].min()} - {df['iteration'].max()}")

if df is not None:
    print("Add your custom analysis here...")

print("Ready for configuration updates...")


def run_quick_analysis(start_col=None, end_col=None):
    quick_config = Config()
    if start_col:
        quick_config.update(start_collection=start_col)
    if end_col:
        quick_config.update(end_collection=end_col)

    quick_analyzer = OptimizationAnalyzer(quick_config)
    quick_analyzer.run_full_analysis()

    quick_df = quick_analyzer.get_dataframe()
    if quick_df is not None:
        print(f"📊 Quick analysis: {len(quick_df)} data points")
        print(quick_analyzer.get_summary())
        return quick_df
    return None


def create_all_plots(dataframe, config_obj):
    if dataframe is None:
        print("No data available for plotting")
        return {}

    plots = {
        "histogram": viz.create_histogram_plot(dataframe, config_obj),
        "boxplot": viz.create_box_plot(dataframe, config_obj),
        "evolution": viz.create_iteration_plot(dataframe, config_obj),
        "scatter": viz.create_scatter_plot(dataframe, config_obj),
    }

    print(f"📊 Created {len(plots)} plot types")
    return plots


print("✅ Interactive analysis script loaded!")
print("💡 Tip: Run cells individually using Ctrl+Enter or use 'Run Cell' in VS Code")

# %%
