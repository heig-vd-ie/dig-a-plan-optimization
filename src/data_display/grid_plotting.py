import pandapower as pp
import polars as pl
from polars import col as c
from general_function import pl_to_dict
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pipelines import DigAPlan, DigAPlanADMM


def plot_grid_from_pandapower(
    net: pp.pandapowerNet,
    dig_a_plan: DigAPlan,
    node_size: int = 22,
    width: int = 800,
    height: int = 700,
    from_z: bool = False,
) -> None:

    if from_z and isinstance(dig_a_plan, DigAPlanADMM):
        switch_status = pl_to_dict(
            dig_a_plan.model_manager.zδ_variable.select("eq_fk", "closed")
        )
    else:
        switch_status = pl_to_dict(
            dig_a_plan.result_manager.extract_switch_status().select(
                "eq_fk", ~c("open")
            )
        )
    net["switch"]["closed"] = net["switch"]["name"].apply(lambda x: switch_status[x])
    bus: pl.DataFrame = pl.from_pandas(net["bus"])

    switch_mapping = {True: ["1.0", "green", "solid"], False: ["0.3", "red", "dash"]}

    bus: pl.DataFrame = pl.from_pandas(net["bus"])
    line: pl.DataFrame = pl.from_pandas(net["line"])
    trafo: pl.DataFrame = pl.from_pandas(net["trafo"])
    switch: pl.DataFrame = pl.from_pandas(net["switch"])
    switch = switch.with_columns(
        c("closed")
        .replace_strict(switch_mapping, default=None)
        .list.to_struct(fields=["opacity", "color", "dash"])
        .alias("config"),
    ).unnest("config")

    load_id = list(net.load["bus"])
    bus = bus.with_columns(
        pl.when(c("bus_id").is_in(load_id))
        .then(pl.lit("circle"))
        .otherwise(pl.lit("square"))
        .alias("symbol")
    )

    fig = go.Figure()

    for data in line.filter(c("max_i_ka") > 5e-2).to_dicts():

        fig.add_trace(
            go.Scatter(
                x=data["x_coords"],
                y=data["y_coords"],
                mode="lines",
                line=dict(width=3, color="blue"),
                hoverinfo="none",
                showlegend=False,
            )
        )
    for data in line.filter(c("max_i_ka") <= 5e-2).to_dicts():

        fig.add_trace(
            go.Scatter(
                x=data["x_coords"],
                y=data["y_coords"],
                mode="lines",
                line=dict(width=3, color="darkviolet"),
                hoverinfo="none",
                showlegend=False,
            )
        )

    for data in trafo.to_dicts():
        fig.add_trace(
            go.Scatter(
                x=data["x_coords"],
                y=data["y_coords"],
                mode="lines",
                line=dict(width=3, color="maroon"),
                hoverinfo="none",
                showlegend=False,
            )
        )

    # draw switchable lines: closed = green solid
    for data in switch.to_dicts():
        fig.add_trace(
            go.Scatter(
                x=data["x_coords"],
                y=data["y_coords"],
                mode="lines",
                opacity=float(data["opacity"]),
                line=dict(width=3, color=data["color"], dash=data["dash"]),
                hoverinfo="none",
                showlegend=False,
            )
        )

    switch_id_mapping = pl_to_dict(
        dig_a_plan.data_manager.edge_data["eq_fk", "edge_id"]
    )
    switch_id = switch.select(
        c("x_coords", "y_coords").list.mean(),
        "color",
        c("name").replace_strict(switch_id_mapping, default=None).alias("edge_id"),
    )
    fig.add_trace(
        go.Scatter(
            x=switch_id["x_coords"].to_list(),
            y=switch_id["y_coords"].to_list(),
            text=switch_id.with_columns("</b>" + c("edge_id").cast(pl.Utf8) + "</b>")[
                "edge_id"
            ].to_list(),
            mode="markers+text",
            hoverinfo="none",
            showlegend=False,
            textfont=dict(color=switch_id["color"].to_list()),
            marker=dict(size=node_size, color="white", symbol="square"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=bus.select(c("coords").list.get(0))["coords"].to_list(),
            y=bus.select(c("coords").list.get(1))["coords"].to_list(),
            text=bus.with_columns("</b>" + c("bus_id").cast(pl.Utf8) + "</b>")[
                "bus_id"
            ].to_list(),
            mode="markers+text",
            hoverinfo="none",
            showlegend=False,
            textfont=dict(color="white"),
            marker=dict(size=node_size, color="blue", symbol=bus["symbol"].to_list()),
        )
    )

    fig.update_layout(
        margin=dict(t=5, l=65, r=10, b=5),
        width=width,  # Set the width of the figure
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    fig.show()


import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_interactive_plot():
    data = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.01,
        row_titles=["Slave objective", "Master objective"],
    )
    data.add_trace(
        go.Scatter(go.Scatter(y=[]), mode="lines", name="Slave objective"), row=1, col=1
    )
    data.add_trace(
        go.Scatter(go.Scatter(y=[]), mode="lines", name="Master objective"),
        row=2,
        col=1,
    )
    data.update_layout(height=400, width=600, margin=dict(t=10, l=20, r=10, b=10))

    return go.FigureWidget(data)
