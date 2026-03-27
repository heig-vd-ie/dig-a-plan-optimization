from experiments import *
import polars as pl
import re
import plotly.graph_objects as go
import plotly.express as px

log = generate_log(name=__name__)

CACHE_FOLDER = "boisy-full"
FORCE = False


def collect_data(startswith_word: str, col: str, force: bool = FORCE) -> pl.DataFrame:
    """Collect data from json files"""
    parquet_file = (
        Path(PROJECT_ROOT)
        / settings.cache.outputs_benchmark
        / CACHE_FOLDER
        / f"{startswith_word}.parquet"
    )
    if parquet_file.exists() and (not force):
        return pl.read_parquet(parquet_file)

    pattern = re.compile(rf"{startswith_word}_(\w+)_(\d{{4}})_+(\w+)\.json")

    files = list(
        (Path(PROJECT_ROOT) / settings.cache.outputs_benchmark / CACHE_FOLDER).glob(
            f"{startswith_word}_*.json"
        )
    )
    dfs = []

    for f in files:
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

    results = pl.concat(dfs, how="vertical", rechunk=True)
    log.info(results)
    results.write_parquet(parquet_file)
    return results


def plot_histogram(
    df: pl.DataFrame,
    variable_column: str,
    target_column: str,
    xaxis_title: str,
    variable_title: str,
):
    df = df.select([variable_column, target_column])
    variables = df[variable_column].unique().sort().to_list()

    colors = px.colors.qualitative.Set2

    fig = go.Figure()
    for i, y in enumerate(variables):
        data = df.filter(pl.col(variable_column) == y)[target_column].to_numpy()

        fig.add_trace(
            go.Histogram(
                x=data,
                name=str(y),
                nbinsx=60,
                opacity=0.65,
                marker=dict(
                    color=colors[i % len(colors)],
                    line=dict(width=0.5, color="black"),
                ),
            )
        )

    fig.update_layout(
        barmode="overlay",
        template="simple_white",
        xaxis=dict(
            title=xaxis_title,
            ticks="outside",
            showgrid=True,
            gridcolor="lightgray",
        ),
        yaxis=dict(
            title="Frequency", ticks="outside", showgrid=True, gridcolor="lightgray"
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
        font=dict(size=14),
        margin=dict(l=60, r=30, t=80, b=60),
    )
    fig.write_html(
        Path(PROJECT_ROOT)
        / settings.cache.figures
        / f"{CACHE_FOLDER}_{target_column}_vs_{variable_column}.html"
    )
    fig.show()


if __name__ == "__main__":
    df = collect_data("congested_lines", "loading_percent")
    plot_histogram(df, "y", "loading_percent", "Loading percent (%)", "Year")
