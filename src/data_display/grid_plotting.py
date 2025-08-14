from typing import List, Tuple
import pandapower as pp
import polars as pl
from polars import col as c
from general_function import pl_to_dict
import plotly.graph_objects as go
import numpy as np
import plotly.colors as pc
import networkx as nx
import igraph as ig

from pipelines.reconfiguration import DigAPlan, DigAPlanADMM


def _create_rainbow_cm(values: list[float], min_val: float, max_val: float) -> List:
    """Create a rainbow colormap for the given values."""
    normalized = (np.array(values) - min_val) / (max_val - min_val)
    return [pc.sample_colorscale("jet", [norm_val])[0] for norm_val in normalized]


def _generate_bus_coordinates(dig_a_plan: DigAPlan, switch_status: dict) -> dict:
    """Generate coordinates for each bus in the network."""
    open_switches = [k for k, v in switch_status.items() if not v]
    edge_data = dig_a_plan.data_manager.edge_data.filter(
        ~((c("type") == "switch") & c("eq_fk").is_in(open_switches))
    )

    external_node = dig_a_plan.data_manager.node_data.filter(
        c("type") == "slack"
    ).get_column("node_id")[0]

    nx_graph = nx.Graph()
    for edge in edge_data.iter_rows(named=True):
        nx_graph.add_edge(edge["u_of_edge"], edge["v_of_edge"])
    for n in nx_graph.nodes():
        nx_graph.nodes[n]["name"] = n

    i_graph = ig.Graph.from_networkx(nx_graph)
    root_idx = i_graph.vs.find(name=external_node).index
    distances = i_graph.shortest_paths(root_idx)[0]
    layers = [-int(d) for d in distances]
    layout = i_graph.layout_sugiyama(layers=layers)
    coord_mapping = dict(zip(range(len(layout)), list(layout)))

    return {
        node_id: coord_mapping[i]
        for i, node_id in enumerate(nx_graph.nodes())
        if i < len(coord_mapping)
    }


def _extract_switch_status(
    net: pp.pandapowerNet, dig_a_plan: DigAPlan, from_z: bool
) -> dict:
    """Extract the switch status from the given pandapower network and DigAPlan."""
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


def _prepare_data_frames_with_algorithm(
    net: pp.pandapowerNet,
    dig_a_plan: DigAPlan,
    switch_status: dict,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Prepare data frames for visualization with the given algorithm."""
    coord_mapping = _generate_bus_coordinates(dig_a_plan, switch_status)

    bus = pl.from_pandas(net["bus"])
    coords_list = [
        coord_mapping[bus_id] if bus_id in coord_mapping else [0, 0]
        for bus_id in bus["bus_id"]
    ]

    bus = bus.with_columns(pl.Series(coords_list).alias("coords"))

    line = pl.from_pandas(net["line"])
    line_x_coords = [
        [coord_mapping.get(from_bus, [0, 0])[0], coord_mapping.get(to_bus, [0, 0])[0]]
        for from_bus, to_bus in zip(line["from_bus"], line["to_bus"])
    ]
    line_y_coords = [
        [coord_mapping.get(from_bus, [0, 0])[1], coord_mapping.get(to_bus, [0, 0])[1]]
        for from_bus, to_bus in zip(line["from_bus"], line["to_bus"])
    ]
    line = line.with_columns(
        pl.Series(line_x_coords).alias("x_coords"),
        pl.Series(line_y_coords).alias("y_coords"),
    )

    trafo = pl.from_pandas(net["trafo"]).join(
        dig_a_plan.data_manager.edge_data.filter(c("type") == "transformer")[
            ["eq_fk", "edge_id"]
        ].rename({"eq_fk": "name"}),
        on="name",
        how="left",
    )
    trafo_x_coords = [
        [coord_mapping.get(hv_bus, [0, 0])[0], coord_mapping.get(lv_bus, [0, 0])[0]]
        for hv_bus, lv_bus in zip(trafo["hv_bus"], trafo["lv_bus"])
    ]
    trafo_y_coords = [
        [coord_mapping.get(hv_bus, [0, 0])[1], coord_mapping.get(lv_bus, [0, 0])[1]]
        for hv_bus, lv_bus in zip(trafo["hv_bus"], trafo["lv_bus"])
    ]

    trafo = trafo.with_columns(
        pl.Series(trafo_x_coords).alias("x_coords"),
        pl.Series(trafo_y_coords).alias("y_coords"),
    )

    switch = pl.from_pandas(net["switch"])
    switch = switch.filter(c("closed"))

    switch_x_coords = [
        [coord_mapping.get(bus_id, [0, 0])[0], coord_mapping.get(element, [0, 0])[0]]
        for bus_id, element in zip(switch["bus"], switch["element"])
    ]
    switch_y_coords = [
        [coord_mapping.get(bus_id, [0, 0])[1], coord_mapping.get(element, [0, 0])[1]]
        for bus_id, element in zip(switch["bus"], switch["element"])
    ]
    switch = switch.with_columns(
        pl.Series(switch_x_coords).alias("x_coords"),
        pl.Series(switch_y_coords).alias("y_coords"),
    )

    return bus, line, trafo, switch


def _prepare_data_frames(
    net: pp.pandapowerNet,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Prepare data frames for visualization."""
    bus = pl.from_pandas(net["bus"])
    line = pl.from_pandas(net["line"])
    trafo = pl.from_pandas(net["trafo"])
    switch = pl.from_pandas(net["switch"])

    return bus, line, trafo, switch


def _get_results_data(
    dig_a_plan: DigAPlan, color_by_results: bool
) -> tuple[dict, dict, dict, dict, dict]:
    """Extract voltage, current, and edge ID mapping data from the DigAPlan."""
    voltage_dict = {}
    current_dict = {}
    edge_id_mapping = {}
    power_dict = {}
    q_dict = {}

    if color_by_results:
        voltages = dig_a_plan.result_manager.extract_node_voltage()
        voltage_dict = pl_to_dict(voltages.select("node_id", "v_pu"))

        currents = dig_a_plan.result_manager.extract_edge_current()
        currents = (
            currents.with_columns([pl.col("p_flow").abs(), pl.col("q_flow").abs()])
            .group_by("edge_id")
            .agg(
                [
                    pl.mean("i_pu").alias("i_pu"),
                    pl.mean("p_flow").alias("p_flow"),
                    pl.mean("q_flow").alias("q_flow"),
                ]
            )
        )
        edge_id_mapping = pl_to_dict(
            dig_a_plan.data_manager.edge_data.select("eq_fk", "edge_id")
        )
        current_dict = pl_to_dict(currents.select("edge_id", "i_pu"))
        power_dict = pl_to_dict(currents.select("edge_id", "p_flow"))
        q_dict = pl_to_dict(currents.select("edge_id", "q_flow"))

    return voltage_dict, current_dict, edge_id_mapping, power_dict, q_dict


def _get_line_currents(
    line: pl.DataFrame,
    trafo: pl.DataFrame,
    edge_id_mapping: dict,
    current_dict: dict,
    power_dict: dict,
    q_dict: dict,
) -> Tuple[
    list[float], list[float], list[float], list[float], list[float], list[float]
]:
    """Extract line currents from the current dictionary using edge ID mapping."""
    line_currents = []
    line_powers = []
    line_qs = []
    trafo_currents = []
    trafo_powers = []
    trafo_qs = []
    for _, row in line.to_pandas().iterrows():
        line_name = row["name"]
        if line_name in edge_id_mapping:
            edge_id = edge_id_mapping[line_name]
            current_val = current_dict.get(edge_id, 0.0)
            power_val = power_dict.get(edge_id, 0.0)
            q_val = q_dict.get(edge_id, 0.0)
        else:
            current_val = 0.0
            power_val = 0.0
            q_val = 0.0
        line_currents.append(current_val)
        line_powers.append(power_val)
        line_qs.append(q_val)
    for _, row in trafo.to_pandas().iterrows():
        trafo_name = row["name"]
        if trafo_name in edge_id_mapping:
            edge_id = edge_id_mapping[trafo_name]
            current_val = current_dict.get(edge_id, 0.0)
            power_val = power_dict.get(edge_id, 0.0)
            q_val = q_dict.get(edge_id, 0.0)
        else:
            current_val = 0.0
            power_val = 0.0
            q_val = 0.0
        trafo_currents.append(current_val)
        trafo_powers.append(power_val)
        trafo_qs.append(q_val)
    return line_currents, line_powers, line_qs, trafo_currents, trafo_powers, trafo_qs


def _add_hover_marker(
    fig: go.Figure,
    x_coords: list,
    y_coords: list,
    hover_text: str,
    name: str | None = None,
):
    """Add a hover marker to the figure."""
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
    return fig


def _plot_colored_lines(
    fig: go.Figure,
    line: pl.DataFrame,
    line_currents: list[float],
    line_powers: list[float],
    line_qs: list[float],
    line_colors: list[str],
):
    """Plot colored lines on the figure."""
    for i, data in enumerate(line.to_dicts()):
        x_coords = data.get("x_coords", [])
        y_coords = data.get("y_coords", [])

        color = line_colors[i] if i < len(line_colors) else "black"

        hover_text = (
            f"Line: {data.get('name', 'N/A')}<br>"
            f"Current: {line_currents[i]:.3f} p.u.<br>"
            f"Power: {line_powers[i]:.3f} p.u.<br>"
            f"Reactive Power: {line_qs[i]:.3f} p.u.<br>"
            f"Max Current: {data.get('max_i_ka', 0):.3f} kA<br>"
            f"Length: {data.get('length_km', 0):.3f} km"
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

        _add_hover_marker(
            fig,
            x_coords,
            y_coords,
            hover_text,
            f"Line {data.get('name', 'N/A')}",
        )


def _plot_default_lines(fig: go.Figure, line: pl.DataFrame, color: str):
    """Plot default lines on the figure."""
    for data in line.to_dicts():
        x_coords = data.get("x_coords", [])
        y_coords = data.get("y_coords", [])

        hover_text = (
            f"Line: {data.get('name', 'N/A')}<br>"
            f"Max Current: {data.get('max_i_ka', 0):.3f} kA<br>"
            f"Length: {data.get('length_km', 0):.3f} km<br>"
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

        _add_hover_marker(fig, x_coords, y_coords, hover_text)


def _plot_default_trafo(fig: go.Figure, trafo: pl.DataFrame, color: str):
    """Plot default transformers on the figure."""
    for data in trafo.to_dicts():
        x_coords = data.get("x_coords", [])
        y_coords = data.get("y_coords", [])

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
                line=dict(width=3, color=color),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        mid_x = (x_coords[0] + x_coords[-1]) / 2
        mid_y = (y_coords[0] + y_coords[-1]) / 2
        fig.add_trace(
            go.Scatter(
                x=[mid_x],
                y=[mid_y],
                mode="text",
                text=["<b>⧚</b>"],
                textfont=dict(size=20, color=color, family="Arial Black"),
                hoverinfo="text",
                hovertext=hover_text,
                showlegend=False,
            )
        )


def _plot_colored_trafo(
    fig: go.Figure,
    trafo: pl.DataFrame,
    trafo_currents: list[float],
    trafo_powers: list[float],
    trafo_qs: list[float],
    trafo_colors: list[str],
):
    """Plot colored transformers on the figure."""
    for i, data in enumerate(trafo.to_dicts()):
        x_coords = data.get("x_coords", [])
        y_coords = data.get("y_coords", [])

        color = trafo_colors[i] if i < len(trafo_colors) else "black"

        hover_text = (
            f"Transformer: {data.get('name', 'N/A')}<br>"
            f"Current: {trafo_currents[i]} p.u.<br>"
            f"Power: {trafo_powers[i]} p.u.<br>"
            f"Reactive Power: {trafo_qs[i]} p.u.<br>"
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
                line=dict(width=3, color=color),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        mid_x = (x_coords[0] + x_coords[-1]) / 2
        mid_y = (y_coords[0] + y_coords[-1]) / 2
        fig.add_trace(
            go.Scatter(
                x=[mid_x],
                y=[mid_y],
                mode="text",
                text=["<b>⧚</b>"],
                textfont=dict(size=20, color=color, family="Arial Black"),
                hoverinfo="text",
                hovertext=hover_text,
                showlegend=False,
            )
        )


def _plot_switches_with_status(fig: go.Figure, switch: pl.DataFrame):
    """Plot switches with their status on the figure."""
    for data in switch.to_dicts():
        x_coords = data.get("x_coords", [])
        y_coords = data.get("y_coords", [])

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


def _plot_buses(
    fig: go.Figure,
    bus: pl.DataFrame,
    node_size: int,
    voltage_dict: dict,
    color_by_results: bool,
    voltage_range: Tuple,
):
    """
    Plot the buses on the grid.
    """
    if color_by_results and voltage_dict:
        bus_voltages = []
        bus_ids = bus["bus_id"].to_list()
        for bus_id in bus_ids:
            voltage_val = voltage_dict.get(bus_id, 1.0)
            bus_voltages.append(voltage_val)

        # Use provided voltage range or calculate from data
        bus_colors = _create_rainbow_cm(
            bus_voltages, min_val=voltage_range[0], max_val=voltage_range[1]
        )

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
                    symbol="circle",
                    line=dict(color="black", width=0),
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
                marker=dict(size=node_size, color="#B6FB8E", symbol="circle"),
            )
        )


def _plot_loads(fig: go.Figure, bus: pl.DataFrame, net: pp.pandapowerNet):
    """
    Plot the loads on the grid.
    """
    load_id = list(net.load["bus"])
    gen_id = list(net.gen["bus"]) + list(net.sgen["bus"])
    for bus_id in load_id:
        bus_row = bus.filter(pl.col("bus_id") == bus_id)
        if len(bus_row) > 0:
            bus_coords = bus_row.select("coords").to_series()[0]
            x_coord = bus_coords[0]
            y_coord = bus_coords[1]

            # Add annotation with rotation for load symbol
            fig.add_annotation(
                x=x_coord + 0.1,
                y=y_coord - 0.4,
                text="⍗",
                font=dict(size=16, color="black", family="Arial Black"),
                textangle=-35,
                showarrow=False,
                hovertext=f"Load at Bus {bus_id}",
            )
    for bus_id in gen_id:
        bus_row = bus.filter(pl.col("bus_id") == bus_id)
        if len(bus_row) > 0:
            bus_coords = bus_row.select("coords").to_series()[0]
            x_coord = bus_coords[0]
            y_coord = bus_coords[1]

            # Add annotation with rotation for load symbol
            fig.add_annotation(
                x=x_coord - 0.1,
                y=y_coord - 0.4,
                text="⍐",
                font=dict(size=16, color="black", family="Arial Black"),
                textangle=35,
                showarrow=False,
                hovertext=f"PV at Bus {bus_id}",
            )


def _add_color_legend(
    fig: go.Figure,
    color_by_results: bool,
    voltage_range: Tuple,
    current_range: Tuple,
):
    """
    Add a color legend to the figure.
    """
    if color_by_results:
        min_volt, max_volt = voltage_range
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(
                    colorscale="jet",
                    showscale=True,
                    cmin=min_volt,
                    cmax=max_volt,
                    colorbar=dict(
                        title=dict(text="Voltage (p.u.)", side="right"),
                        x=1.02,
                        y=0.8,
                        len=0.3,
                        thickness=15,
                    ),
                ),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Add current colorbar
        min_curr, max_curr = current_range
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(
                    colorscale="jet",
                    showscale=True,
                    cmin=min_curr,
                    cmax=max_curr,
                    colorbar=dict(
                        title=dict(text="Current (p.u.)", side="right"),
                        x=1.02,
                        y=0.4,
                        len=0.3,
                        thickness=15,
                    ),
                ),
                showlegend=False,
                hoverinfo="skip",
            )
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
    """Plot the grid from a pandapower network."""
    switch_status = _extract_switch_status(net, dig_a_plan, from_z)

    if color_by_results:
        bus, line, trafo, switch = _prepare_data_frames_with_algorithm(
            net, dig_a_plan, switch_status
        )
    else:
        bus, line, trafo, switch = _prepare_data_frames(net)

    voltage_dict, current_dict, edge_id_mapping, power_dict, q_dict = _get_results_data(
        dig_a_plan, color_by_results
    )

    fig = go.Figure()

    voltage_range = (
        dig_a_plan.data_manager.node_data["v_min_pu"].min(),
        dig_a_plan.data_manager.node_data["v_max_pu"].max(),
    )

    if color_by_results:
        line_currents, line_powers, line_qs, trafo_currents, trafo_powers, trafo_qs = (
            _get_line_currents(
                line=line,
                trafo=trafo,
                edge_id_mapping=edge_id_mapping,
                current_dict=current_dict,
                power_dict=power_dict,
                q_dict=q_dict,
            )
        )
        colors = _create_rainbow_cm(
            line_currents + trafo_currents,
            min_val=min(line_currents + trafo_currents),
            max_val=max(line_currents + trafo_currents),
        )
        line_colors = colors[: len(line_currents)]
        trafo_colors = colors[len(line_currents) :]
        all_currents = line_currents + trafo_currents

        _plot_colored_lines(
            fig,
            line=line,
            line_currents=line_currents,
            line_powers=line_powers,
            line_qs=line_qs,
            line_colors=line_colors,
        )
        _plot_colored_trafo(
            fig,
            trafo=trafo,
            trafo_powers=trafo_powers,
            trafo_currents=trafo_currents,
            trafo_qs=trafo_qs,
            trafo_colors=trafo_colors,
        )
        _plot_switches_with_status(fig, switch)
        current_range = (min(all_currents), max(all_currents))
        _add_color_legend(fig, color_by_results, voltage_range, current_range)
    else:
        _plot_default_lines(fig, line, "black")
        _plot_default_trafo(fig, trafo, "black")
        _plot_switches_with_status(fig, switch)

    _plot_buses(fig, bus, node_size, voltage_dict, color_by_results, voltage_range)
    _plot_loads(fig, bus, net)

    fig.update_layout(
        margin=dict(t=5, l=65, r=120, b=5),  # Increased right margin for colorbars
        width=width,
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    fig.show()
    fig.write_html(
        f".cache/grid_plot{'_colored' if color_by_results else '_default'}.html",
        include_plotlyjs="cdn",
    )
