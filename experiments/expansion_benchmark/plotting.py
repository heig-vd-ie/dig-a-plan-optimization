from typing import Literal

from experiments import *
import polars as pl
import re
import plotly.graph_objects as go
import plotly.express as px
import argparse
from tqdm import tqdm

log = generate_log(name=__name__)


def collect_data(
    CACHE_FOLDER: str, startswith_word: str, col: str, force: bool
) -> pl.DataFrame:
    """Collect data from json files"""
    if not force:
        files = list(
            (Path(PROJECT_ROOT) / settings.cache.outputs_benchmark / CACHE_FOLDER).glob(
                f"{startswith_word}_*.parquet"
            )
        )
        dfs: list[pl.DataFrame] = []
        for f in tqdm(files, desc="Loading files"):
            df = pl.read_parquet(f)
            dfs.append(df)

        return pl.concat(dfs, how="vertical")

    pattern = re.compile(rf"{startswith_word}_(\w+)_(\d{{4}})_+(\w+)\.json")

    files = list(
        (Path(PROJECT_ROOT) / settings.cache.outputs_benchmark / CACHE_FOLDER).glob(
            f"{startswith_word}_*.json"
        )
    )
    log.info(f"Collect data from {len(files)} different files!")
    dfs: list[pl.DataFrame] = []

    for f in tqdm(files, desc="Loading files"):
        match = pattern.search(f.name)
        if not match:
            continue

        scenario = str(match.group(1))
        year = int(match.group(2))
        time_step = str(match.group(3))

        df = pl.read_json(f)

        if col not in df.columns:
            continue

        lp = df.get_column(col)[0]
        if lp is None:
            continue

        tmp = pl.DataFrame(
            {
                "scenario": [scenario] * len(lp),
                "y": [year] * len(lp),
                "t": [time_step] * len(lp),
                "id": list(lp.keys()),
                col: list(lp.values()),
            }
        ).with_columns(
            pl.col("scenario").cast(pl.Utf8),
            pl.col("y").cast(pl.Int64),
            pl.col("t").cast(pl.Utf8),
            pl.col("id").cast(pl.Int32),
            pl.col(col).cast(pl.Float64),
        )

        tmp.write_parquet(
            Path(PROJECT_ROOT)
            / settings.cache.outputs_benchmark
            / CACHE_FOLDER
            / f"{startswith_word}_{scenario}_{year}_{time_step}.parquet"
        )

        dfs.append(tmp)

    if not dfs:
        return pl.DataFrame(
            schema={
                "scenario": pl.Utf8,
                "y": pl.Int32,
                "t": pl.Utf8,
                "id": pl.Int32,
                col: pl.Float64,
            }
        )

    results = pl.concat(dfs, how="vertical")
    log.info(f"Collected {results.shape[0]} rows of data for {startswith_word}!")
    return results


def plot_histogram(
    CACHE_FOLDER: str,
    df: pl.DataFrame,
    variable_column: str,
    target_column: str,
    xaxis_title: str,
    variable_title: str,
):
    df = df.select([variable_column, target_column])

    grouped = (
        df.group_by(variable_column)
        .agg(pl.col(target_column).alias("values"))
        .sort(variable_column)
    )

    variables = grouped[variable_column].to_list()
    values_list = grouped["values"].to_list()

    colors = px.colors.qualitative.Set2

    all_values = np.concatenate([np.asarray(v) for v in values_list])
    bin_edges = np.histogram_bin_edges(all_values, bins=100)

    fig = go.Figure()

    for i, (y, data) in enumerate(zip(variables, values_list)):
        data = np.asarray(data)

        hist, _ = np.histogram(data, bins=bin_edges)

        fig.add_trace(
            go.Bar(
                x=bin_edges[:-1],
                y=hist,
                name=str(y),
                opacity=0.65,
                marker=dict(color=colors[i % len(colors)]),
            )
        )

    fig.update_layout(
        barmode="overlay",
        template="simple_white",
        xaxis=dict(
            title=xaxis_title,
            showgrid=True,
            gridcolor="lightgray",
            ticks="outside",
        ),
        yaxis=dict(
            title="Frequency",
            showgrid=True,
            gridcolor="lightgray",
            ticks="outside",
        ),
        legend=dict(
            title=variable_title,
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        width=900,
        height=550,
    )

    fig.write_html(
        Path(PROJECT_ROOT)
        / settings.cache.figures
        / f"{CACHE_FOLDER}_{target_column}_vs_{variable_column}.html"
    )

    fig.show()


def calculate_expansion_cost(CACHE_FOLDER, FORCE, edge_type: Literal["line", "trafo"]):
    df_edge = collect_data(CACHE_FOLDER, f"expanded_{edge_type}s", "delta", FORCE)

    with open(
        Path(PROJECT_ROOT)
        / settings.cache.outputs_benchmark
        / CACHE_FOLDER
        / "input.json"
    ) as f:
        req = json.load(f)

    if edge_type == "line":
        net = pp.from_pickle(req["grid"]["pp_file"])
        lines_pd = net.line
        buses_pd = net.bus
        lines_pd["id"] = lines_pd.index
        buses_pd["id"] = buses_pd.index
        lines = pl.from_dataframe(lines_pd)[["id", "length_km", "from_bus"]]
        buses = pl.from_dataframe(buses_pd)[["id", "vn_kv"]]
        df_edge = df_edge.join(lines, on="id", how="left").join(
            buses, left_on="from_bus", right_on="id", how="left"
        )
        df_edge = df_edge.select(
            pl.col("id"),
            pl.col("y"),
            (pl.col("delta") * pl.col("length_km") * pl.col("vn_kv") * 1000).alias(
                "delta"
            ),
        )
    df_edge = df_edge.with_columns(
        [
            (
                (pl.col("delta") * (1000 if edge_type == "trafo" else 1))
                * req["congestion_settings"][
                    (
                        "line_cost_per_km_kw"
                        if edge_type == "line"
                        else "trafo_cost_per_kw"
                    )
                ]
            ).alias("cost")
        ]
    )
    df_edge = df_edge.with_columns(
        [
            (
                pl.col("cost")
                * (
                    (1 - req["congestion_settings"]["discount_rate"])
                    ** (pl.col("y").cast(pl.Float64) - 2025)
                )
            ).alias("net_present_cost")
        ]
    )
    df_edge.write_parquet(
        Path(PROJECT_ROOT)
        / settings.cache.outputs_benchmark
        / CACHE_FOLDER
        / f"expanded_{edge_type}s.parquet"
    )
    log.info(
        f"Calculated expansion cost for {edge_type}s is {df_edge['net_present_cost'].sum() / 1000000:.4f} million CHF."
    )
    return df_edge


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--CACHE_FOLDER", type=str, required=True)
    parser.add_argument("--FORCE", action="store_true", help="Force recomputation")

    args = parser.parse_args()

    df = collect_data(
        args.CACHE_FOLDER, "congested_lines", "loading_percent", args.FORCE
    )
    plot_histogram(
        args.CACHE_FOLDER, df, "y", "loading_percent", "Loading percent (%)", "Year"
    )
    df = collect_data(
        args.CACHE_FOLDER, "congested_trafos", "loading_percent", args.FORCE
    )
    plot_histogram(
        args.CACHE_FOLDER, df, "y", "loading_percent", "Loading percent (%)", "Year"
    )
    df = collect_data(args.CACHE_FOLDER, "ou_buses", "vm_pu", args.FORCE)
    plot_histogram(args.CACHE_FOLDER, df, "y", "vm_pu", "Voltage (p.u.)", "Year")

    line_cost = calculate_expansion_cost(
        args.CACHE_FOLDER, args.FORCE, edge_type="line"
    )
    trafo_cost = calculate_expansion_cost(
        args.CACHE_FOLDER, args.FORCE, edge_type="trafo"
    )
