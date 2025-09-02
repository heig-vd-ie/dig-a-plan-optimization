#!/usr/bin/env python3

import sys
import os

# Ensure we're in the plots directory for local imports
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

from config import Config
from analyzer import OptimizationAnalyzer
import visualizer as viz
import data_processor as dp


def quick_example():
    print("🚀 Quick Example: Optimization Results Analysis")

    config = Config(
        start_collection="run_20250902_110438", end_collection="run_20250902_130518"
    )

    analyzer = OptimizationAnalyzer(config)
    analyzer.run_full_analysis()

    df = analyzer.get_dataframe()
    if df is None:
        print("❌ No data available")
        return

    print(f"📊 Loaded {len(df)} data points")

    summary = analyzer.get_summary()
    print("\n📈 Summary:")
    print(summary)

    fig = viz.create_histogram_plot(df, config)
    fig.show()

    risk_methods = df["risk_method"].unique()
    if len(risk_methods) > 1:
        first_method = risk_methods[0]
        filtered_df = dp.filter_data_by_risk_method(df, first_method)
        print(f"\n🔍 Filtered data for {first_method}: {len(filtered_df)} points")


def advanced_example():
    print("\n🔬 Advanced Example: Custom Analysis")

    config = Config()
    analyzer = OptimizationAnalyzer(config)

    try:
        df = (
            analyzer.connect()
            .load_collections()
            .extract_risk_info()
            .process_data()
            .get_dataframe()
        )

        if df is not None:
            plots = {
                "histogram": viz.create_histogram_plot(df, config),
                "boxplot": viz.create_box_plot(df, config),
                "evolution": viz.create_iteration_plot(df, config),
                "scatter": viz.create_scatter_plot(df, config),
            }

            print(f"📊 Generated {len(plots)} plot types")

            plots["histogram"].show()

    except Exception as e:
        print(f"❌ Advanced example failed: {e}")


if __name__ == "__main__":
    try:
        quick_example()
        advanced_example()
        print("\n✅ Examples completed!")

    except Exception as e:
        print(f"❌ Example failed: {e}")
        print("Make sure MongoDB is running and collections exist.")
