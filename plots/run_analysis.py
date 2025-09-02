#!/usr/bin/env python3

import sys
import os
import argparse

# Ensure we're in the plots directory for local imports
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

from config import Config
from analyzer import OptimizationAnalyzer
import visualizer as viz


def main():
    parser = argparse.ArgumentParser(description="Analyze optimization results")
    parser.add_argument("--start", help="Start collection name")
    parser.add_argument("--end", help="End collection name")
    parser.add_argument("--host", default="localhost", help="MongoDB host")
    parser.add_argument("--port", type=int, default=27017, help="MongoDB port")
    parser.add_argument("--db", default="optimization", help="Database name")
    parser.add_argument(
        "--show-plots", action="store_true", help="Show interactive plots"
    )
    parser.add_argument("--save-plots", help="Directory to save plot images")

    args = parser.parse_args()

    config_params = {
        "mongodb_host": args.host,
        "mongodb_port": args.port,
        "database_name": args.db,
    }

    if args.start:
        config_params["start_collection"] = args.start
    if args.end:
        config_params["end_collection"] = args.end

    config = Config(**config_params)

    print("🔧 Configuration:")
    print(
        f"   Database: {config.mongodb_host}:{config.mongodb_port}/{config.database_name}"
    )
    print(f"   Collections: {config.start_collection} to {config.end_collection}")

    print("\n📊 Running analysis...")
    analyzer = OptimizationAnalyzer(config)

    try:
        analyzer.run_full_analysis()
        df = analyzer.get_dataframe()

        if df is None:
            print("❌ No data retrieved from analysis")
            sys.exit(1)

        print(f"✅ Analysis completed!")
        print(f"   Collections loaded: {len(analyzer.collections)}")
        print(f"   Data points: {len(df)}")
        print(f"   Risk methods: {list(df['risk_method'].unique())}")

        print("\n📈 Summary Statistics:")
        summary = analyzer.get_summary()
        print(summary)

        if args.show_plots:
            print("\n📊 Generating interactive plots...")

            fig_hist = viz.create_histogram_plot(df, config)
            fig_box = viz.create_box_plot(df, config)
            fig_line = viz.create_iteration_plot(df, config)
            fig_scatter = viz.create_scatter_plot(df, config)

            fig_hist.show()
            fig_box.show()
            fig_line.show()
            fig_scatter.show()

        if args.save_plots:
            print(f"\n💾 Saving plots to {args.save_plots}...")
            import os

            os.makedirs(args.save_plots, exist_ok=True)

            fig_hist = viz.create_histogram_plot(df, config)
            fig_box = viz.create_box_plot(df, config)
            fig_line = viz.create_iteration_plot(df, config)
            fig_scatter = viz.create_scatter_plot(df, config)

            fig_hist.write_html(f"{args.save_plots}/histogram.html")
            fig_box.write_html(f"{args.save_plots}/boxplot.html")
            fig_line.write_html(f"{args.save_plots}/evolution.html")
            fig_scatter.write_html(f"{args.save_plots}/scatter.html")

            print("   Plots saved as HTML files")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
