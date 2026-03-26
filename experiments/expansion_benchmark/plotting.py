from experiments import *
import polars as pl

if __name__ == "__main__":
    cache_folder = (
        Path(PROJECT_ROOT) / settings.cache.outputs_combined / "boisy-feeder-1"
    )
    df = (
        pl.scan_ndjson(cache_folder / "congested_lines_*.json", include_file_paths=True)
        .with_columns(
            pl.col("file_path")
            .str.extract(r"congested_lines_(\w+)_(\d{4})__([0-9]+)", 1)
            .alias("scenario"),
            pl.col("file_path")
            .str.extract(r"congested_lines_(\w+)_(\d{4})__([0-9]+)", 2)
            .cast(pl.Int32)
            .alias("year"),
            pl.col("file_path")
            .str.extract(r"congested_lines_(\w+)_(\d{4})__([0-9]+)", 3)
            .cast(pl.Int32)
            .alias("time_step"),
        )
        .drop("file_path")
        .collect()
    )
