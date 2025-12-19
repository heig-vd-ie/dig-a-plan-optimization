import os
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import polars as pl
from polars import col as c
import plotly.graph_objects as go
import plotly.colors as pc
from plotly.subplots import make_subplots
import plotly.colors
from PIL import ImageColor
from _plotly_utils.basevalidators import ColorscaleValidator
import igraph as ig
from helper_functions import pl_to_dict, generate_tree_graph_from_edge_data
from data_schema import NodeEdgeModel
from pipelines.reconfiguration import DigAPlan, DigAPlanADMM
from data_display.style import apply_plot_style

# -----------------------------
# Data containers
# -----------------------------


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


# -----------------------------
# Legacy colorscale utilities
# -----------------------------


def _create_rainbow_cm(values: list[float], min_val: float, max_val: float) -> List:
    """Create a rainbow colormap for the given values."""
    if np.isclose(max_val, min_val):
        max_val = min_val + 1e-12
    normalized = (np.array(values) - min_val) / (max_val - min_val)
    normalized = np.clip(normalized, 0.0, 1.0)
    return [pc.sample_colorscale("jet", [norm_val])[0] for norm_val in normalized]


# -----------------------------
# Extraction helpers
# -----------------------------


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


def _prepare_dfs_w_coordinates(dap: DigAPlan, switch_status: dict) -> GridDataFrames:
    """
    Prepare bus/edge frames with x_coords/y_coords lists for plotting.

    If is_generate_coords is True:
        Generate bus coordinates from the energized topology (tree layout).

    If is_generate_coords is False:
        Use existing coordinates stored in net.bus["coords"].
    """
    bus = dap.data_manager.node_data

    edges = dap.data_manager.edge_data

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
    show_switch_ids: bool = False,
    edge_id_mapping: Optional[dict] = None,  # eq_fk -> edge_id
):
    """
    Plot switches with their status on the figure.
    Added from legacy: optional edge_id labels at the switch midpoint.
    """
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
        if show_switch_ids and edge_id_mapping is not None:
            eq_fk = data.get("eq_fk", None)
            edge_id = edge_id_mapping.get(eq_fk) if eq_fk is not None else None
            if edge_id is not None:
                text_color = "green" if is_closed else "red"
                fig.add_trace(
                    go.Scatter(
                        x=[mid_x],
                        y=[mid_y],
                        text=[f"</b>{edge_id}</b>"],
                        mode="markers+text",
                        hoverinfo="skip",
                        showlegend=False,
                        textfont=dict(color=text_color),
                        marker=dict(size=node_size, color="white", symbol="square"),
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


def _plot_loads(fig: go.Figure, grid_dfs: GridDataFrames, node_size: int):
    """
    Plot the loads on the grid.
    """
    load_id = (
        grid_dfs.bus.filter(pl.col("cons_installed") > 1e-3)
        .get_column("node_id")
        .to_list()
    )
    gen_id = (
        grid_dfs.bus.filter(pl.col("prod_installed") > 1e-3)
        .get_column("node_id")
        .to_list()
    )
    for node_id in load_id:
        bus_row = grid_dfs.bus.filter(pl.col("node_id") == node_id)
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
        bus_row = grid_dfs.bus.filter(pl.col("node_id") == node_id)
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
    node_size: int = 22,
    text_size: int = 10,
    width: int = 800,
    height: int = 700,
    from_z: bool = False,
    color_by_results: bool = False,
    line_default_color: str = "black",
    trafo_default_color: str = "black",
    show_switch_ids: bool = False,
) -> None:
    """Plot the grid from a pandapower network."""
    switch_status = _extract_switch_status(dap, from_z)
    grid_dfs = _prepare_dfs_w_coordinates(dap, switch_status)
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
        dap.data_manager.node_data["min_vm_pu"].min(),
        dap.data_manager.node_data["max_vm_pu"].max(),
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
            show_switch_ids=show_switch_ids,
            edge_id_mapping=result_data.edge_id_mapping,
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
            fig,
            grid_dfs.switches,
            plot_open_switch=True,
            node_size=node_size,
            show_switch_ids=show_switch_ids,
            edge_id_mapping=result_data.edge_id_mapping,
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

    _plot_loads(fig, grid_dfs=grid_dfs, node_size=node_size)

    title = "Grid Topology (Colored Flows)" if color_by_results else "Grid Topology"
    apply_plot_style(
        fig,
        x_title="",
        y_title="",
        title=title,
    )

    fig.update_layout(
        margin=dict(t=5, l=65, r=120, b=5),
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
    cv = ColorscaleValidator("colorscale", "")
    colorscale = cv.validate_coerce(colorscale_name)
    if hasattr(loc, "__iter__"):
        return [get_continuous_color(colorscale, x) for x in loc]
    return get_continuous_color(colorscale, loc)


def get_continuous_color(colorscale, intermed):
    if len(colorscale) < 1:
        raise ValueError("colorscale must have at least one color")

    # Ensure float and clamp to [0, 1]
    intermed = float(intermed)
    if intermed <= 0 or len(colorscale) == 1:
        return (
            colorscale[0][1]
            if colorscale[0][1][0] != "#"
            else "rgb" + str(ImageColor.getcolor(colorscale[0][1], "RGB"))
        )
    if intermed >= 1:
        return (
            colorscale[-1][1]
            if colorscale[-1][1][0] != "#"
            else "rgb" + str(ImageColor.getcolor(colorscale[-1][1], "RGB"))
        )

    hex_to_rgb = lambda cc: "rgb" + str(ImageColor.getcolor(cc, "RGB"))

    low_cutoff, low_color = colorscale[0]
    high_cutoff, high_color = colorscale[-1]

    for cutoff, color in colorscale:
        if cutoff <= intermed:
            low_cutoff, low_color = cutoff, color
        if cutoff >= intermed:
            high_cutoff, high_color = cutoff, color
            break

    if low_color[0] == "#":
        low_color = hex_to_rgb(low_color)
    if high_color[0] == "#":
        high_color = hex_to_rgb(high_color)

    denom = high_cutoff - low_cutoff
    if denom == 0:
        return low_color

    t = (intermed - low_cutoff) / denom

    return plotly.colors.find_intermediate_color(
        lowcolor=low_color,
        highcolor=high_color,
        intermed=t,
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
    """
    Independent plot used for PF results using NodeEdgeModel + switches/currents/voltages tables.
    """
    fig: go.Figure = go.Figure()

    open_switches_id: pl.Series = switches.filter(c("open"))["edge_id"]
    edge_data: pl.DataFrame = base_grid_data.edge_data.filter(
        ~c("edge_id").is_in(open_switches_id)
    )
    node_data: pl.DataFrame = base_grid_data.node_data
    slack_node_id = base_grid_data.node_data.filter(c("type") == "slack")["node_id"][0]
    nx_tree_grid = generate_tree_graph_from_edge_data(
        edge_data=edge_data, slack_node_id=slack_node_id
    )

    i_graph: ig.Graph = ig.Graph.from_networkx(nx_tree_grid)
    layout = i_graph.layout_reingold_tilford()
    coord_mapping = dict(zip(nx_tree_grid.nodes, list(layout)))  # type: ignore

    symbol_mapping = {"slack": "diamond", "pq": "circle"}

    node_data = (
        node_data.join(voltages, on="node_id", how="left")
        .with_columns(
            c("node_id").replace_strict(coord_mapping, default=None).alias("coord"),
            c("type").replace_strict(symbol_mapping, default=None).alias("symbol"),
        )
        .with_columns(
            c("coord").list.get(0).alias("x"),
            (-c("coord").list.get(1)).alias("y"),
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

    if max_loading is None:
        max_loading = max(100, float(edge_data["loading"].max()))  # type: ignore

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
                            edge_colorscale, float(data["loading"]) / float(max_loading)
                        ),
                    ),
                    hoverinfo="none",
                    showlegend=False,
                )
            )

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
                size=node_size,
                color=line_data["loading"],
                colorscale=edge_colorscale,
                showscale=True,
                cmin=0,
                cmax=max_loading,
                colorbar=dict(title="Edge loading [%]", x=1.2),
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
            marker=dict(opacity=0, size=node_size, color=switch_color),
            showlegend=False,
        )
    )

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
                size=node_size,
                color=node_data["v_pu"],
                colorscale=node_colorscale,
                showscale=True,
                cmin=min_voltage,
                cmax=max_voltage,
                colorbar=dict(title="Node voltage [pu]"),
            ),
            showlegend=False,
        )
    )

    apply_plot_style(fig, x_title="", y_title="", title="Power Flow Results")

    fig.update_layout(
        margin=dict(t=10, l=5, r=5, b=5),
        width=width,
        height=height,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)

    os.makedirs(".cache/figs", exist_ok=True)
    fig.write_html(".cache/figs/boisy_grid_plot.html", include_plotlyjs="cdn")
    return fig
