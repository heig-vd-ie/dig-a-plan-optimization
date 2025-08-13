import pandapower as pp
import polars as pl
from polars import col as c
from general_function import pl_to_dict
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import plotly.colors as pc
import networkx as nx
import igraph as ig

from pipelines.reconfiguration import DigAPlan, DigAPlanADMM


def create_rainbow_colormap(
    values: list[float], min_val: float, max_val: float
) -> list[str]:
    if not values:
        return []

    values_array = np.array(values)
    actual_min = min_val if min_val is not None else values_array.min()
    actual_max = max_val if max_val is not None else values_array.max()

    if actual_max == actual_min:
        return ["rgb(255, 0, 0)"] * len(values)

    normalized = (values_array - actual_min) / (actual_max - actual_min)
    colors = []
    for norm_val in normalized:
        color = pc.sample_colorscale("rainbow", [norm_val])[0]
        colors.append(color)

    return colors


def generate_bus_coordinates(dig_a_plan: DigAPlan, switch_status: dict) -> dict:
    open_switches = [k for k, v in switch_status.items() if not v]
    edge_data = dig_a_plan.data_manager.edge_data.filter(
        ~((c("type") == "switch") & c("eq_fk").is_in(open_switches))
    )
    nx_graph = nx.Graph()
    for edge in edge_data.iter_rows(named=True):
        nx_graph.add_edge(edge["u_of_edge"], edge["v_of_edge"])

    i_graph = ig.Graph.from_networkx(nx_graph)
    layout = i_graph.layout_sugiyama()
    coord_mapping = dict(zip(range(len(layout)), list(layout)))

    node_mapping = {}
    for i, node_id in enumerate(nx_graph.nodes()):
        if i < len(coord_mapping):
            node_mapping[node_id] = coord_mapping[i]

    return node_mapping


def extract_switch_status(
    net: pp.pandapowerNet, dig_a_plan: DigAPlan, from_z: bool
) -> dict:
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
    return switch_status


def prepare_data_frames_with_algorithm(
    net: pp.pandapowerNet,
    dig_a_plan: DigAPlan,
    switch_status: dict,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    coord_mapping = generate_bus_coordinates(dig_a_plan, switch_status)

    bus = pl.from_pandas(net["bus"])
    coords_list = []
    for bus_id in bus["bus_id"]:
        if bus_id in coord_mapping:
            coords_list.append(coord_mapping[bus_id])
        else:
            coords_list.append([0, 0])

    bus = bus.with_columns(pl.Series(coords_list).alias("coords"))

    line = pl.from_pandas(net["line"])
    line_x_coords = []
    line_y_coords = []
    for from_bus, to_bus in zip(line["from_bus"], line["to_bus"]):
        from_coord = coord_mapping.get(from_bus, [0, 0])
        to_coord = coord_mapping.get(to_bus, [0, 0])
        line_x_coords.append([from_coord[0], to_coord[0]])
        line_y_coords.append([from_coord[1], to_coord[1]])

    line = line.with_columns(
        pl.Series(line_x_coords).alias("x_coords"),
        pl.Series(line_y_coords).alias("y_coords"),
    )

    trafo = pl.from_pandas(net["trafo"])
    trafo_x_coords = []
    trafo_y_coords = []
    for hv_bus, lv_bus in zip(trafo["hv_bus"], trafo["lv_bus"]):
        hv_coord = coord_mapping.get(hv_bus, [0, 0])
        lv_coord = coord_mapping.get(lv_bus, [0, 0])
        trafo_x_coords.append([hv_coord[0], lv_coord[0]])
        trafo_y_coords.append([hv_coord[1], lv_coord[1]])

    trafo = trafo.with_columns(
        pl.Series(trafo_x_coords).alias("x_coords"),
        pl.Series(trafo_y_coords).alias("y_coords"),
    )

    switch = pl.from_pandas(net["switch"])
    switch = switch.filter(c("closed"))

    switch_x_coords = []
    switch_y_coords = []
    for bus_id, element in zip(switch["bus"], switch["element"]):
        bus_coord = coord_mapping.get(bus_id, [0, 0])
        element_coord = coord_mapping.get(element, [0, 0])
        switch_x_coords.append([bus_coord[0], element_coord[0]])
        switch_y_coords.append([bus_coord[1], element_coord[1]])

    switch = switch.with_columns(
        pl.Series(switch_x_coords).alias("x_coords"),
        pl.Series(switch_y_coords).alias("y_coords"),
    )

    load_id = list(net.load["bus"])
    bus = bus.with_columns(
        pl.when(c("bus_id").is_in(load_id))
        .then(pl.lit("circle"))
        .otherwise(pl.lit("circle"))
        .alias("symbol")
    )

    return bus, line, trafo, switch


def prepare_data_frames(
    net: pp.pandapowerNet,
    dig_a_plan: DigAPlan,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    bus = pl.from_pandas(net["bus"])
    line = pl.from_pandas(net["line"])
    trafo = pl.from_pandas(net["trafo"])
    switch = pl.from_pandas(net["switch"])

    load_id = list(net.load["bus"])
    bus = bus.with_columns(
        pl.when(c("bus_id").is_in(load_id))
        .then(pl.lit("circle"))
        .otherwise(pl.lit("circle"))
        .alias("symbol")
    )

    return bus, line, trafo, switch


def get_results_data(
    dig_a_plan: DigAPlan, color_by_results: bool
) -> tuple[dict, dict, dict]:
    voltage_dict = {}
    current_dict = {}
    edge_id_mapping = {}

    if color_by_results:
        voltages = dig_a_plan.result_manager.extract_node_voltage()
        voltage_dict = pl_to_dict(voltages.select("node_id", "v_pu"))

        currents = dig_a_plan.result_manager.extract_edge_current()
        currents = currents.filter(pl.col("from_node_id") > pl.col("to_node_id"))
        edge_id_mapping = pl_to_dict(
            dig_a_plan.data_manager.edge_data.select("eq_fk", "edge_id")
        )
        current_dict = pl_to_dict(currents.select("edge_id", "i_pct"))

    return voltage_dict, current_dict, edge_id_mapping


def get_line_currents(
    line: pl.DataFrame, edge_id_mapping: dict, current_dict: dict
) -> list[float]:
    line_currents = []
    for _, row in line.to_pandas().iterrows():
        line_name = row["name"]
        if line_name in edge_id_mapping:
            edge_id = edge_id_mapping[line_name]
            current_val = current_dict.get(edge_id, 0.0) * 100
        else:
            current_val = 0.0
        line_currents.append(current_val)
    return line_currents


def add_hover_marker(
    fig: go.Figure,
    x_coords: list,
    y_coords: list,
    hover_text: str,
    name: str | None = None,
):
    if len(x_coords) >= 2 and len(y_coords) >= 2:
        mid_x = (x_coords[0] + x_coords[-1]) / 2
        mid_y = (y_coords[0] + y_coords[-1]) / 2
        fig.add_trace(
            go.Scatter(
                x=[mid_x],
                y=[mid_y],
                mode="markers",
                marker=dict(size=12, color="rgba(0,0,0,0)"),
                hoverinfo="text",
                hovertext=hover_text,
                showlegend=False,
                name=name,
            )
        )


def plot_colored_lines(
    fig: go.Figure,
    line: pl.DataFrame,
    line_currents: list[float],
    line_colors: list[str],
):
    for i, data in enumerate(line.to_dicts()):
        x_coords = data.get("x_coords", [])
        y_coords = data.get("y_coords", [])

        if not x_coords or not y_coords or len(x_coords) != len(y_coords):
            continue

        line_width = 3 if data.get("max_i_ka", 0) > 5e-2 else 2
        color = line_colors[i] if i < len(line_colors) else "blue"

        hover_text = (
            f"Line: {data.get('name', 'N/A')}<br>"
            f"Current: {line_currents[i]:.3f} %<br>"
            f"Max Current: {data.get('max_i_ka', 0):.3f} kA<br>"
            f"Length: {data.get('length_km', 0):.3f} km"
        )

        fig.add_trace(
            go.Scatter(
                x=x_coords,
                y=y_coords,
                mode="lines",
                line=dict(width=line_width, color=color),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        add_hover_marker(
            fig,
            x_coords,
            y_coords,
            hover_text,
            f"Line {data.get('name', 'N/A')}",
        )


def plot_default_lines(
    fig: go.Figure,
    line: pl.DataFrame,
    capacity_filter: str,
    color: str,
    line_type: str,
):
    filter_condition = (
        c("max_i_ka") > 5e-2 if capacity_filter == "high" else c("max_i_ka") <= 5e-2
    )

    for data in line.filter(filter_condition).to_dicts():
        x_coords = data.get("x_coords", [])
        y_coords = data.get("y_coords", [])

        if not x_coords or not y_coords or len(x_coords) != len(y_coords):
            continue

        hover_text = (
            f"Line: {data.get('name', 'N/A')}<br>"
            f"Max Current: {data.get('max_i_ka', 0):.3f} kA<br>"
            f"Length: {data.get('length_km', 0):.3f} km<br>"
            f"Type: {line_type}"
        )

        fig.add_trace(
            go.Scatter(
                x=x_coords,
                y=y_coords,
                mode="lines",
                line=dict(width=3, color=color),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        add_hover_marker(fig, x_coords, y_coords, hover_text)


def plot_transformers(fig: go.Figure, trafo: pl.DataFrame):
    for data in trafo.to_dicts():
        x_coords = data.get("x_coords", [])
        y_coords = data.get("y_coords", [])

        if not x_coords or not y_coords or len(x_coords) != len(y_coords):
            continue

        hover_text = (
            f"Transformer: {data.get('name', 'N/A')}<br>"
            f"HV Bus: {data.get('hv_bus', 'N/A')}<br>"
            f"LV Bus: {data.get('lv_bus', 'N/A')}<br>"
            f"Power Rating: {data.get('sn_mva', 0):.2f} MVA<br>"
            f"HV Voltage: {data.get('vn_hv_kv', 0):.1f} kV<br>"
            f"LV Voltage: {data.get('vn_lv_kv', 0):.1f} kV"
        )

        fig.add_trace(
            go.Scatter(
                x=x_coords,
                y=y_coords,
                mode="lines",
                line=dict(width=3, color="maroon"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        if len(x_coords) >= 2 and len(y_coords) >= 2:
            mid_x = (x_coords[0] + x_coords[-1]) / 2
            mid_y = (y_coords[0] + y_coords[-1]) / 2
            fig.add_trace(
                go.Scatter(
                    x=[mid_x],
                    y=[mid_y],
                    mode="text",
                    text=["⌭"],
                    textfont=dict(size=20, color="maroon", family="Arial Black"),
                    hoverinfo="text",
                    hovertext=hover_text,
                    showlegend=False,
                )
            )


def plot_switches_with_status(fig: go.Figure, switch: pl.DataFrame):
    for data in switch.to_dicts():
        x_coords = data.get("x_coords", [])
        y_coords = data.get("y_coords", [])

        if not x_coords or not y_coords or len(x_coords) != len(y_coords):
            continue

        is_closed = data.get("closed", False)
        status_text = "Closed" if is_closed else "Open"
        hover_text = (
            f"Switch: {data.get('name', 'N/A')}<br>"
            f"Status: {status_text}<br>"
            f"From Bus: {data.get('bus', 'N/A')}<br>"
            f"To Element: {data.get('element', 'N/A')}<br>"
            f"Element Type: {data.get('et', 'N/A')}"
        )

        line_style = "solid" if is_closed else "dot"
        fig.add_trace(
            go.Scatter(
                x=x_coords,
                y=y_coords,
                mode="lines",
                line=dict(width=3, color="black", dash=line_style),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        if len(x_coords) >= 2 and len(y_coords) >= 2:
            mid_x = (x_coords[0] + x_coords[-1]) / 2
            mid_y = (y_coords[0] + y_coords[-1]) / 2
            box_color = "black" if is_closed else "white"
            box_line_color = "black"

            fig.add_trace(
                go.Scatter(
                    x=[mid_x],
                    y=[mid_y],
                    mode="markers",
                    marker=dict(
                        size=8,
                        color=box_color,
                        symbol="square",
                        line=dict(color=box_line_color, width=1),
                    ),
                    hoverinfo="text",
                    hovertext=hover_text,
                    showlegend=False,
                )
            )


def plot_switches(fig: go.Figure, switch: pl.DataFrame):
    for data in switch.to_dicts():
        x_coords = data.get("x_coords", [])
        y_coords = data.get("y_coords", [])

        if not x_coords or not y_coords or len(x_coords) != len(y_coords):
            continue

        hover_text = (
            f"Switch: {data.get('name', 'N/A')}<br>"
            f"Status: Closed<br>"
            f"From Bus: {data.get('bus', 'N/A')}<br>"
            f"To Element: {data.get('element', 'N/A')}<br>"
            f"Element Type: {data.get('et', 'N/A')}"
        )

        fig.add_trace(
            go.Scatter(
                x=x_coords,
                y=y_coords,
                mode="lines",
                line=dict(width=3, color="black", dash="solid"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        if len(x_coords) >= 2 and len(y_coords) >= 2:
            mid_x = (x_coords[0] + x_coords[-1]) / 2
            mid_y = (y_coords[0] + y_coords[-1]) / 2

            fig.add_trace(
                go.Scatter(
                    x=[mid_x],
                    y=[mid_y],
                    mode="markers",
                    marker=dict(
                        size=8,
                        color="black",
                        symbol="square",
                        line=dict(color="black", width=1),
                    ),
                    hoverinfo="text",
                    hovertext=hover_text,
                    showlegend=False,
                )
            )


def plot_buses(
    fig: go.Figure,
    bus: pl.DataFrame,
    node_size: int,
    voltage_dict: dict,
    color_by_results: bool,
):
    if color_by_results and voltage_dict:
        bus_voltages = []
        bus_ids = bus["bus_id"].to_list()
        for bus_id in bus_ids:
            voltage_val = voltage_dict.get(bus_id, 1.0)
            bus_voltages.append(voltage_val)

        bus_colors = create_rainbow_colormap(bus_voltages, min_val=0.95, max_val=1.05)

        fig.add_trace(
            go.Scatter(
                x=bus.select(c("coords").list.get(0))["coords"].to_list(),
                y=bus.select(c("coords").list.get(1))["coords"].to_list(),
                text=bus.with_columns("</b>" + c("bus_id").cast(pl.Utf8) + "</b>")[
                    "bus_id"
                ].to_list(),
                mode="markers+text",
                hoverinfo="text",
                hovertext=[
                    f"Bus {bus_id}<br>Voltage: {volt:.3f} pu"
                    for bus_id, volt in zip(bus_ids, bus_voltages)
                ],
                showlegend=False,
                textfont=dict(color="black"),
                marker=dict(
                    size=node_size,
                    color=bus_colors,
                    symbol=bus["symbol"].to_list(),
                    line=dict(color="black", width=1),
                ),
            )
        )
    else:
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
                textfont=dict(color="black"),
                marker=dict(
                    size=node_size, color="cyan", symbol=bus["symbol"].to_list()
                ),
            )
        )


def plot_loads(fig: go.Figure, bus: pl.DataFrame, net: pp.pandapowerNet):
    load_id = list(net.load["bus"])
    for bus_id in load_id:
        bus_row = bus.filter(pl.col("bus_id") == bus_id)
        if len(bus_row) > 0:
            bus_coords = bus_row.select("coords").to_series()[0]
            x_coord = bus_coords[0]
            y_coord = bus_coords[1]

            fig.add_trace(
                go.Scatter(
                    x=[x_coord],
                    y=[y_coord - 0.5],
                    mode="text",
                    text=["⏃"],
                    textfont=dict(size=16, color="black", family="Arial Black"),
                    hoverinfo="text",
                    hovertext=f"Load at Bus {bus_id}",
                    showlegend=False,
                )
            )


def add_color_legend(fig: go.Figure, color_by_results: bool):
    if color_by_results:
        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref="paper",
            yref="paper",
            text="<b>Color Legend:</b><br>"
            + "Nodes: Voltage (Blue=Low, Red=High)<br>"
            + "Lines: Current (Blue=Low, Red=High)",
            showarrow=False,
            font=dict(size=10),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1,
        )


def plot_grid_from_pandapower(
    net: pp.pandapowerNet,
    dig_a_plan: DigAPlan,
    node_size: int = 22,
    width: int = 800,
    height: int = 700,
    from_z: bool = False,
    color_by_results: bool = False,
) -> None:
    switch_status = extract_switch_status(net, dig_a_plan, from_z)

    if color_by_results:
        bus, line, trafo, switch = prepare_data_frames_with_algorithm(
            net, dig_a_plan, switch_status
        )
    else:
        bus, line, trafo, switch = prepare_data_frames(net, dig_a_plan)

    voltage_dict, current_dict, edge_id_mapping = get_results_data(
        dig_a_plan, color_by_results
    )

    fig = go.Figure()

    if color_by_results and current_dict:
        line_currents = get_line_currents(line, edge_id_mapping, current_dict)
        if line_currents:
            line_colors = create_rainbow_colormap(
                line_currents,
                min_val=min(line_currents) if line_currents else 0.0,
                max_val=max(line_currents) if line_currents else 1.0,
            )
            plot_colored_lines(fig, line, line_currents, line_colors)
    else:
        plot_default_lines(fig, line, "high", "blue", "High Capacity")
        plot_default_lines(fig, line, "low", "darkviolet", "Low Capacity")

    plot_transformers(fig, trafo)

    if color_by_results:
        plot_switches(fig, switch)
    else:
        plot_switches_with_status(fig, switch)

    plot_buses(fig, bus, node_size, voltage_dict, color_by_results)
    plot_loads(fig, bus, net)
    add_color_legend(fig, color_by_results)

    fig.update_layout(
        margin=dict(t=5, l=65, r=10, b=5),
        width=width,
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        title="Grid Topology"
        + (" with Color-Coded Results" if color_by_results else ""),
    )
    fig.show()
