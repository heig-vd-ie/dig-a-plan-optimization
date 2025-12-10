import os
from dataclasses import dataclass
from typing import List, Tuple
import pandapower as pp
import polars as pl
from polars import col as c
from helper_functions import pl_to_dict
import plotly.graph_objects as go
import numpy as np
import plotly.colors as pc
import networkx as nx
import igraph as ig
from pipelines.reconfiguration import DigAPlan, DigAPlanADMM
from data_display.style import apply_plot_style


@dataclass
class GridDataFrames:
    bus: pl.DataFrame
    line: pl.DataFrame
    trafo: pl.DataFrame
    switches: pl.DataFrame


@dataclass
class ResultData:
    voltage_dict: dict
    current_dict: dict
    p_dict: dict
    q_dict: dict
    edge_id_mapping: dict


@dataclass
class EdgeFlows:
    p: list[float]
    q: list[float]
    i: list[float]


@dataclass
class EdgeFlowResults:
    line: EdgeFlows
    trafo: EdgeFlows


def _create_rainbow_cm(values: list[float], min_val: float, max_val: float) -> List:
    """Create a rainbow colormap for the given values."""
    normalized = (np.array(values) - min_val) / (max_val - min_val)
    return [pc.sample_colorscale("jet", [norm_val])[0] for norm_val in normalized]


def _generate_bus_coordinates(dap: DigAPlan, switch_status: dict) -> dict:
    """Generate coordinates for each bus in the network."""
    open_switches = [k for k, v in switch_status.items() if not v]
    edge_data = dap.data_manager.edge_data.filter(
        ~((c("type") == "switch") & c("eq_fk").is_in(open_switches))
    )

    external_node = dap.data_manager.node_data.filter(c("type") == "slack").get_column(
        "node_id"
    )[0]

    nx_graph = nx.Graph()
    for edge in edge_data.iter_rows(named=True):
        nx_graph.add_edge(edge["u_of_edge"], edge["v_of_edge"])
    for n in nx_graph.nodes():
        nx_graph.nodes[n]["node_id"] = n

    i_graph = ig.Graph.from_networkx(nx_graph)
    root_idx = i_graph.vs.find(node_id=external_node).index
    distances = i_graph.shortest_paths(root_idx)[0]
    layers = [-int(d) for d in distances]
    layout = i_graph.layout_sugiyama(layers=layers)
    coord_mapping = dict(zip(range(len(layout)), list(layout)))

    return {
        node_id: coord_mapping[i]
        for i, node_id in enumerate(nx_graph.nodes())
        if i < len(coord_mapping)
    }


def _extract_switch_status(dap: DigAPlan, from_z: bool) -> dict:
    """Extract the switch status from the given pandapower network and DigAPlan."""
    if from_z and isinstance(dap, DigAPlanADMM):
        switch_status = pl_to_dict(
            dap.model_manager.zδ_variable.select("eq_fk", "closed")
        )
    else:
        switch_status = pl_to_dict(
            dap.result_manager.extract_switch_status().select("eq_fk", ~c("open"))
        )
    return switch_status


def _prepare_dfs_w_coordinates(
    dap: DigAPlan,
    net: pp.pandapowerNet,
    switch_status: dict,
    is_generate_coords: bool = False,
) -> GridDataFrames:
    """Prepare data frames for visualization with the given algorithm."""
    bus = dap.data_manager.node_data

    edges = dap.data_manager.edge_data
    if is_generate_coords:
        coord_mapping = _generate_bus_coordinates(dap, switch_status)
        coord_mapping_pl = pl.DataFrame(
            {
                "node_id": list(coord_mapping.keys()),
                "coords": [
                    [float(coords[0]), float(coords[1])]
                    for coords in coord_mapping.values()
                ],
            }
        )

    else:
        coord_mapping_pl = pl.DataFrame(
            {
                "node_id": list(net.bus.index),
                "coords": [
                    [float(coords[0]), float(coords[1])]
                    for coords in net.bus["coords"].values.tolist()
                ],
            }
        )

    bus = bus.join(coord_mapping_pl, on="node_id", how="left")

    edges = (
        edges.join(
            coord_mapping_pl.rename({"node_id": "u_of_edge", "coords": "from_coords"}),
            on="u_of_edge",
            how="left",
        )
        .join(
            coord_mapping_pl.rename({"node_id": "v_of_edge", "coords": "to_coords"}),
            on="v_of_edge",
            how="left",
        )
        .with_columns(
            [
                pl.concat_list(
                    [c("from_coords").list.get(0), c("to_coords").list.get(0)]
                ).alias("x_coords"),
                pl.concat_list(
                    [c("from_coords").list.get(1), c("to_coords").list.get(1)]
                ).alias("y_coords"),
            ]
        )
    )
    line = edges.filter(c("type") == "branch")
    trafo = edges.filter(c("type") == "transformer")
    switches = edges.filter(c("type") == "switch")
    switch_status_pl = pl.DataFrame(
        {
            "eq_fk": list(switch_status.keys()),
            "closed": list(switch_status.values()),
        }
    )
    switches = switches.join(switch_status_pl, on="eq_fk", how="left")

    return GridDataFrames(bus=bus, line=line, trafo=trafo, switches=switches)


def _get_results_data(dap: DigAPlan) -> ResultData:
    """Extract voltage, current, and edge ID mapping data from the DigAPlan."""
    voltages = dap.result_manager.extract_node_voltage()
    voltage_dict = pl_to_dict(voltages.select("node_id", "v_pu"))

    currents = dap.result_manager.extract_edge_current()
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
    edge_id_mapping = pl_to_dict(dap.data_manager.edge_data.select("eq_fk", "edge_id"))
    current_dict = pl_to_dict(currents.select("edge_id", "i_pu"))
    p_dict = pl_to_dict(currents.select("edge_id", "p_flow"))
    q_dict = pl_to_dict(currents.select("edge_id", "q_flow"))

    return ResultData(
        voltage_dict=voltage_dict,
        current_dict=current_dict,
        p_dict=p_dict,
        q_dict=q_dict,
        edge_id_mapping=edge_id_mapping,
    )


def _get_edge_flows(
    grid_data: GridDataFrames, result_data: ResultData
) -> EdgeFlowResults:
    """Extract line currents from the current dictionary using edge ID mapping."""
    line_is = []
    line_ps = []
    line_qs = []
    trafo_is = []
    trafo_ps = []
    trafo_qs = []
    for _, row in grid_data.line.to_pandas().iterrows():
        line_name = row["eq_fk"]
        edge_id = result_data.edge_id_mapping[line_name]
        i_val = result_data.current_dict.get(edge_id, 0.0)
        p_val = result_data.p_dict.get(edge_id, 0.0)
        q_val = result_data.q_dict.get(edge_id, 0.0)
        line_is.append(i_val)
        line_ps.append(p_val)
        line_qs.append(q_val)
    for _, row in grid_data.trafo.to_pandas().iterrows():
        trafo_name = row["eq_fk"]
        edge_id = result_data.edge_id_mapping[trafo_name]
        i_val = result_data.current_dict.get(edge_id, 0.0)
        p_val = result_data.p_dict.get(edge_id, 0.0)
        q_val = result_data.q_dict.get(edge_id, 0.0)
        trafo_is.append(i_val)
        trafo_ps.append(p_val)
        trafo_qs.append(q_val)
    return EdgeFlowResults(
        line=EdgeFlows(i=line_is, p=line_ps, q=line_qs),
        trafo=EdgeFlows(i=trafo_is, p=trafo_ps, q=trafo_qs),
    )


def _add_hover_marker(
    fig: go.Figure,
    x_coords: list,
    y_coords: list,
    hover_text: str,
    name: str | None = None,
):
    """Add a hover marker to the figure."""
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


def _plot_colored_lines(
    fig: go.Figure,
    grid_data: GridDataFrames,
    line_flows: EdgeFlows,
    line_colors: list[str],
    default_color: str,
    node_size: float,
):
    """Plot colored lines on the figure."""
    for i, data in enumerate(grid_data.line.to_dicts()):
        x_coords = data.get("x_coords", [])
        y_coords = data.get("y_coords", [])

        color = line_colors[i] if i < len(line_colors) else default_color

        hover_text = (
            f"Line: {data.get('eq_fk', 'N/A')}<br>"
            f"Current: {line_flows.i[i]:.3f} p.u.<br>"
            f"Power: {line_flows.p[i]:.3f} p.u.<br>"
            f"Reactive Power: {line_flows.q[i]:.3f} p.u.<br>"
            f"Max Current: {data.get('i_max_pu', 0):.3f} p.u."
        )

        fig.add_trace(
            go.Scatter(
                x=x_coords,
                y=y_coords,
                mode="lines",
                line=dict(width=3 * node_size / 22, color=color),
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


def _plot_default_lines(
    fig: go.Figure, line: pl.DataFrame, default_color: str, node_size: float
):
    """Plot default lines on the figure."""
    for data in line.to_dicts():
        x_coords = data.get("x_coords", [])
        y_coords = data.get("y_coords", [])

        hover_text = (
            f"Line: {data.get('name', 'N/A')}<br>"
            f"Max Current: {data.get('i_max_pu', 0):.3f} p.u."
        )

        fig.add_trace(
            go.Scatter(
                x=x_coords,
                y=y_coords,
                mode="lines",
                line=dict(width=3 * node_size / 22, color=default_color),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        _add_hover_marker(fig, x_coords, y_coords, hover_text)


def _plot_default_trafo(
    fig: go.Figure, trafo: pl.DataFrame, default_color: str, node_size: float
):
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
                line=dict(width=3 * node_size / 22, color=default_color),
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
                textfont=dict(
                    size=20 * node_size / 22, color=default_color, family="Arial Black"
                ),
                hoverinfo="text",
                hovertext=hover_text,
                showlegend=False,
            )
        )


def _plot_colored_trafo(
    fig: go.Figure,
    grid_data: GridDataFrames,
    trafo_flows: EdgeFlows,
    trafo_colors: list[str],
    default_color: str,
    node_size: float,
):
    """Plot colored transformers on the figure."""
    for i, data in enumerate(grid_data.trafo.to_dicts()):
        x_coords = data.get("x_coords", [])
        y_coords = data.get("y_coords", [])

        color = trafo_colors[i] if i < len(trafo_colors) else default_color

        hover_text = (
            f"Transformer: {data.get('name', 'N/A')}<br>"
            f"Current: {trafo_flows.i[i]} p.u.<br>"
            f"Power: {trafo_flows.p[i]} p.u.<br>"
            f"Reactive Power: {trafo_flows.q[i]} p.u.<br>"
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
                line=dict(width=3 * node_size / 22, color=color),
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
                textfont=dict(
                    size=20 * node_size / 22, color=color, family="Arial Black"
                ),
                hoverinfo="text",
                hovertext=hover_text,
                showlegend=False,
            )
        )


def _plot_switches(
    fig: go.Figure,
    switch: pl.DataFrame,
    node_size: float,
    plot_open_switch: bool = True,
):
    """Plot switches with their status on the figure."""
    for data in switch.to_dicts():
        x_coords = data.get("x_coords", [])
        y_coords = data.get("y_coords", [])

        is_closed = data.get("closed", False)
        if (not plot_open_switch) and (not is_closed):
            continue
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
                line=dict(width=3 * node_size / 22, color="black", dash=line_style),
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
                    size=8 * node_size / 22,
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
    grid_dfs: GridDataFrames,
    result_data: ResultData,
    node_size: int,
    color_by_results: bool,
    voltage_range: Tuple,
    text_size: int,
):
    """
    Plot the buses on the grid.
    """
    if color_by_results and result_data.voltage_dict:
        bus_voltages = []
        node_ids = grid_dfs.bus["node_id"].to_list()
        for node_id in node_ids:
            voltage_val = result_data.voltage_dict.get(node_id, 1.0)
            bus_voltages.append(voltage_val)

        # Use provided voltage range or calculate from data
        bus_colors = _create_rainbow_cm(
            bus_voltages, min_val=voltage_range[0], max_val=voltage_range[1]
        )

        fig.add_trace(
            go.Scatter(
                x=grid_dfs.bus.select(c("coords").list.get(0))["coords"].to_list(),
                y=grid_dfs.bus.select(c("coords").list.get(1))["coords"].to_list(),
                text=grid_dfs.bus.with_columns(
                    "</b>" + c("node_id").cast(pl.Utf8) + "</b>"
                )["node_id"].to_list(),
                mode="markers+text",
                hoverinfo="text",
                hovertext=[
                    f"Bus {node_id}<br>Voltage: {volt:.3f} pu"
                    for node_id, volt in zip(node_ids, bus_voltages)
                ],
                showlegend=False,
                textfont=dict(color="black", size=text_size),
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
                x=grid_dfs.bus.select(c("coords").list.get(0))["coords"].to_list(),
                y=grid_dfs.bus.select(c("coords").list.get(1))["coords"].to_list(),
                text=grid_dfs.bus.with_columns(
                    "</b>" + c("node_id").cast(pl.Utf8) + "</b>"
                )["node_id"].to_list(),
                mode="markers+text",
                hoverinfo="none",
                showlegend=False,
                textfont=dict(color="black"),
                marker=dict(size=node_size, color="#B6FB8E", symbol="circle"),
            )
        )


def _plot_loads(
    fig: go.Figure, net: pp.pandapowerNet, grid_dfs: GridDataFrames, node_size: int
):
    """
    Plot the loads on the grid.
    """
    load_id = list(net.load["bus"])
    bus = grid_dfs.bus
    gen_id = list(net.gen["bus"]) + list(net.sgen["bus"])
    for node_id in load_id:
        bus_row = bus.filter(pl.col("node_id") == node_id)
        if len(bus_row) > 0:
            bus_coords = bus_row.select("coords").to_series()[0]
            x_coord = bus_coords[0]
            y_coord = bus_coords[1]

            # Add annotation with rotation for load symbol
            fig.add_annotation(
                x=x_coord + 0.1 * node_size / 22,
                y=y_coord - 0.4 * node_size / 22,
                text="⍗",
                font=dict(
                    size=node_size * 16 / 22, color="black", family="Arial Black"
                ),
                textangle=-35,
                showarrow=False,
                hovertext=f"Load at Bus {node_id}",
            )
    for node_id in gen_id:
        bus_row = bus.filter(pl.col("node_id") == node_id)
        if len(bus_row) > 0:
            bus_coords = bus_row.select("coords").to_series()[0]
            x_coord = bus_coords[0]
            y_coord = bus_coords[1]

            # Add annotation with rotation for load symbol
            fig.add_annotation(
                x=x_coord - 0.1 * node_size / 22,
                y=y_coord - 0.4 * node_size / 22,
                text="⍐",
                font=dict(
                    size=node_size * 16 / 22, color="black", family="Arial Black"
                ),
                textangle=35,
                showarrow=False,
                hovertext=f"PV at Bus {node_id}",
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
    dap: DigAPlan,
    net: pp.pandapowerNet,
    node_size: int = 22,
    text_size: int = 10,
    width: int = 800,
    height: int = 700,
    from_z: bool = False,
    color_by_results: bool = False,
    line_default_color: str = "black",
    trafo_default_color: str = "black",
) -> None:
    """Plot the grid from a pandapower network."""
    switch_status = _extract_switch_status(dap, from_z)
    grid_dfs = _prepare_dfs_w_coordinates(
        dap, net, switch_status, is_generate_coords=color_by_results
    )
    result_data = _get_results_data(dap)

    fig = go.Figure()

    if color_by_results:
        edge_flows = _get_edge_flows(grid_dfs, result_data)
        colors = _create_rainbow_cm(
            edge_flows.line.i + edge_flows.trafo.i,
            min_val=min(edge_flows.line.i + edge_flows.trafo.i),
            max_val=max(edge_flows.line.i + edge_flows.trafo.i),
        )
        line_colors = colors[: len(edge_flows.line.i)]
        trafo_colors = colors[len(edge_flows.line.i) :]
        all_currents = edge_flows.line.i + edge_flows.trafo.i
        current_range = (min(all_currents), max(all_currents))

    voltage_range = (
        dap.data_manager.node_data["v_min_pu"].min(),
        dap.data_manager.node_data["v_max_pu"].max(),
    )

    if color_by_results:
        _plot_colored_lines(
            fig,
            grid_data=grid_dfs,
            line_flows=edge_flows.line,
            line_colors=line_colors,
            default_color=line_default_color,
            node_size=node_size,
        )
        _plot_colored_trafo(
            fig,
            grid_data=grid_dfs,
            trafo_flows=edge_flows.trafo,
            trafo_colors=trafo_colors,
            default_color=trafo_default_color,
            node_size=node_size,
        )
        _plot_switches(
            fig,
            grid_dfs.switches,
            plot_open_switch=False,
            node_size=node_size,
        )
        _plot_buses(
            fig,
            grid_dfs=grid_dfs,
            result_data=result_data,
            node_size=node_size,
            color_by_results=True,
            voltage_range=voltage_range,
            text_size=text_size,
        )
        _add_color_legend(fig, color_by_results, voltage_range, current_range)
    else:
        _plot_default_lines(fig, grid_dfs.line, line_default_color, node_size=node_size)
        _plot_default_trafo(
            fig, grid_dfs.trafo, trafo_default_color, node_size=node_size
        )
        _plot_switches(
            fig, grid_dfs.switches, plot_open_switch=True, node_size=node_size
        )
        _plot_buses(
            fig,
            grid_dfs=grid_dfs,
            result_data=result_data,
            node_size=node_size,
            color_by_results=False,
            voltage_range=voltage_range,
            text_size=text_size,
        )

    _plot_loads(fig, net=net, grid_dfs=grid_dfs, node_size=node_size)
    
    title = (
        "Grid Topology (Colored Flows)" if color_by_results else "Grid Topology"
    )
    apply_plot_style(
        fig,
        x_title="",
        y_title="",
        title=title,
    )

    fig.update_layout(
        margin=dict(t=5, l=65, r=120, b=5),  # Increased right margin for colorbars
        width=width,
        height=height,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
    
    os.makedirs(".cache/figs", exist_ok=True)
    fig.write_html(
        f".cache/figs/grid_plot{'_colored' if color_by_results else '_default'}.html",
        include_plotlyjs="cdn",
    )
    fig.write_image(
        f".cache/figs/grid_plot{'_colored' if color_by_results else '_default'}.svg",
        format="svg",
    )
