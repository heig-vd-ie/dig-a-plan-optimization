import os
from typing import Optional
import pandapower as pp
import polars as pl
from polars import col as c
from helper_functions import pl_to_dict
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import igraph as ig

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors
from PIL import ImageColor
from _plotly_utils.basevalidators import ColorscaleValidator

from helper_functions import generate_tree_graph_from_edge_data
from data_schema import NodeEdgeModel
from pipelines.reconfiguration import DigAPlan, DigAPlanADMM

from data_display.style import apply_plot_style


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
    
    apply_plot_style(fig, x_title="", y_title="", title="Grid Topology")

    fig.update_layout(
        margin=dict(t=5, l=65, r=10, b=5),
        width=width,  # Set the width of the figure
        height=height,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
    
    fig.show()


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


def get_color(colorscale_name, loc):

    # first parameter: Name of the property being validated
    # second parameter: a string, doesn't really matter in our use case
    cv = ColorscaleValidator("colorscale", "")
    # colorscale will be a list of lists: [[loc1, "rgb1"], [loc2, "rgb2"], ...]
    colorscale = cv.validate_coerce(colorscale_name)

    if hasattr(loc, "__iter__"):
        return [get_continuous_color(colorscale, x) for x in loc]
    return get_continuous_color(colorscale, loc)


# Identical to Adam's answer


def get_continuous_color(colorscale, intermed):
    """
    Plotly continuous colorscales assign colors to the range [0, 1]. This function computes the intermediate
    color for any value in that range.

    Plotly doesn't make the colorscales directly accessible in a common format.
    Some are ready to use:
        colorscale = plotly.colors.PLOTLY_SCALES["Greens"]

    Others are just swatches that need to be constructed into a colorscale:

        viridis_colors, scale = plotly.colors.convert_colors_to_same_type(plotly.colors.sequential.Viridis)
        colorscale = plotly.colors.make_colorscale(viridis_colors, scale=scale)

    :param colorscale: A plotly continuous colorscale defined with RGB string colors.
    :param intermed: value in the range [0, 1]
    :return: color in rgb string format
    :rtype: str
    """
    if len(colorscale) < 1:
        raise ValueError("colorscale must have at least one color")

    hex_to_rgb = lambda c: "rgb" + str(ImageColor.getcolor(c, "RGB"))

    if intermed <= 0 or len(colorscale) == 1:
        c = colorscale[0][1]
        return c if c[0] != "#" else hex_to_rgb(c)
    if intermed >= 1:
        c = colorscale[-1][1]
        return c if c[0] != "#" else hex_to_rgb(c)

    for cutoff, color in colorscale:
        if intermed > cutoff:
            low_cutoff, low_color = cutoff, color
        else:
            high_cutoff, high_color = cutoff, color
            break

    if (low_color[0] == "#") or (high_color[0] == "#"):
        # some color scale names (such as cividis) returns:
        # [[loc1, "hex1"], [loc2, "hex2"], ...]
        low_color = hex_to_rgb(low_color)
        high_color = hex_to_rgb(high_color)

    return plotly.colors.find_intermediate_color(
        lowcolor=low_color,
        highcolor=high_color,
        intermed=((intermed - low_cutoff) / (high_cutoff - low_cutoff)),  # type: ignore
        colortype="rgb",
    )


def plot_power_flow_results(
    base_grid_data: NodeEdgeModel,
    switches: pl.DataFrame,
    currents: pl.DataFrame,
    voltages: pl.DataFrame,
    node_size: int = 10,
    edge_width: int = 3,
    width: int = 1200,
    height: int = 800,
    node_colorscale: str = "Viridis",
    edge_colorscale: str = "Portland",
    switch_color: str = "black",
    min_voltage: Optional[float] = None,
    max_voltage: Optional[float] = None,
    max_loading: Optional[float] = None,
):

    fig: go.Figure = go.Figure()

    # Remove open switch from edge data to get a tree-based grid
    open_switches_id: pl.Series = switches.filter(c("open"))["edge_id"]
    edge_data: pl.DataFrame = base_grid_data.edge_data.filter(
        ~c("edge_id").is_in(open_switches_id)
    )
    node_data: pl.DataFrame = base_grid_data.node_data
    slack_node_id = base_grid_data.node_data.filter(c("type") == "slack")["node_id"][0]
    nx_tree_grid = generate_tree_graph_from_edge_data(
        edge_data=edge_data, slack_node_id=slack_node_id
    )

    # Generate nodes coordinates
    i_graph: ig.Graph = ig.Graph.from_networkx(nx_tree_grid)
    layout = i_graph.layout_reingold_tilford()
    coord_mapping = dict(zip(nx_tree_grid.nodes, list(layout)))  # type: ignore

    symbol_mapping = {"slack": "diamond", "pq": "circle"}

    # Add voltage, loading results and coordinates to grid data
    node_data = (
        node_data.join(voltages, on="node_id", how="left")
        .with_columns(
            c("node_id").replace_strict(coord_mapping, default=None).alias("coord"),
            c("type").replace_strict(symbol_mapping, default=None).alias("symbol"),
        )
        .with_columns(
            c("coord").list.get(0).alias("x"),
            (-c("coord").list.get(1)).alias("y"),  # Put slack node on the top
            pl.concat_list(
                c("node_id").cast(pl.Utf8), c("v_pu").round(3).cast(pl.Utf8)
            ).alias("customdata"),
        )
    )

    edge_data = (
        edge_data.join(currents["edge_id", "i_pu"], on="edge_id", how="left")
        .with_columns(
            pl.concat_list(
                c("u_of_edge", "v_of_edge").replace_strict(coord_mapping, default=None)
            ).alias("coord")
        )
        .with_columns(
            c("coord").list.gather_every(n=2).alias("x"),
            c("coord")
            .list.gather_every(n=2, offset=1)
            .list.eval(pl.element().neg())
            .alias("y"),
            (c("i_pu") / c("i_max_pu") * 100).alias("loading"),
        )
        .with_columns(
            c("x").list.mean().alias("x_mean"),
            c("y").list.mean().alias("y_mean"),
            pl.concat_list(
                c("edge_id").cast(pl.Utf8),
                c("type"),
                c("i_pu").round(3).cast(pl.Utf8),
                c("loading").round(1).cast(pl.Utf8),
            ).alias("customdata"),
        )
    )

    # Plot edge data
    if max_loading is None:
        max_loading = max(100, edge_data["loading"].max())  # type: ignore

    for data in edge_data.to_dicts():
        if data["type"] == "switch":
            fig.add_trace(
                go.Scatter(
                    x=data["x"],
                    y=data["y"],
                    mode="lines",
                    line=dict(width=edge_width, color=switch_color),
                    hoverinfo="none",
                    showlegend=False,
                    opacity=0.5,
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=data["x"],
                    y=data["y"],
                    mode="lines",
                    line=dict(
                        width=edge_width,
                        color=get_color(
                            edge_colorscale, data["loading"] / max_loading
                        ),  # From % to pu
                    ),
                    hoverinfo="none",
                    showlegend=False,
                )
            )

    # Add Hover text in the middle of each edge
    switch_data = edge_data.filter(c("type") == "switch")
    line_data = edge_data.filter(c("type") != "switch")

    hovertemplate = (
        "<b>Edge id:</b> %{customdata[0]}<br>"
        + "<b>Type:</b> %{customdata[1]}<br>"
        + "<b>Current [pu]:</b> %{customdata[2]}<br>"
        + "<b>Loading [%]:</b> %{customdata[3]}<extra></extra>"
    )
    fig.add_trace(
        go.Scatter(
            x=line_data["x_mean"].to_list(),
            y=line_data["y_mean"].to_list(),
            customdata=line_data["customdata"].to_list(),
            hovertemplate=hovertemplate,
            mode="markers",
            marker=dict(
                opacity=0,
                size=node_size,  # Size of the markers
                color=line_data["loading"],  # The third parameter for coloring
                colorscale=edge_colorscale,  # A continuous color scale (e.g., 'Viridis', 'Jet', 'Plasma')
                showscale=True,  # Display the color bar
                cmin=0,
                cmax=max_loading,
                colorbar=dict(
                    title="Edge loading [%]", x=1.2
                ),  # Title for the color bar
            ),
            showlegend=False,
        )
    )

    hovertemplate = "<b>Edge id:</b> %{customdata[0]}<br><b>Type:</b> %{customdata[1]}<extra></extra>"

    fig.add_trace(
        go.Scatter(
            x=switch_data["x_mean"].to_list(),
            y=switch_data["y_mean"].to_list(),
            customdata=switch_data["customdata"].to_list(),
            hovertemplate=hovertemplate,
            mode="markers",
            marker=dict(
                opacity=0,
                size=node_size,  # Size of the markers
                color=switch_color,  # The third parameter for coloring
            ),
            showlegend=False,
        )
    )

    # Plot nodes
    hovertemplate = "<b>Node id:</b> %{customdata[0]}<br><b>Voltage [pu]:</b> %{customdata[1]}<extra></extra>"

    fig.add_trace(
        go.Scatter(
            x=node_data["x"].to_list(),
            y=node_data["y"].to_list(),
            customdata=node_data["customdata"].to_list(),
            hovertemplate=hovertemplate,
            mode="markers",
            marker=dict(
                symbol=node_data["symbol"].to_list(),
                size=node_size,  # Size of the markers
                color=node_data["v_pu"],  # The third parameter for coloring
                colorscale=node_colorscale,  # A continuous color scale (e.g., 'Viridis', 'Jet', 'Plasma')
                showscale=True,  # Display the color bar
                cmin=min_voltage,
                cmax=max_voltage,
                colorbar=dict(title="Node votage [pu]"),  # Title for the color bar
            ),
            showlegend=False,
            # hoverinfo="none",
        )
    )
    apply_plot_style(fig, x_title="", y_title="", title="Power Flow Results")

    fig.update_layout(
        margin=dict(t=10, l=5, r=5, b=5),
        width=width,  # Set the width of the figure
        height=height,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)

    os.makedirs(".cache/figs", exist_ok=True)  
    fig.write_html(".cache/figs/boisy_grid_plot.html", include_plotlyjs="cdn")
    return fig
