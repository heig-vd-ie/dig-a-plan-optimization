import pandapower as pp
import polars as pl
from polars import col as c
from general_function import pl_to_dict
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

from pipelines.reconfiguration import DigAPlan, DigAPlanADMM


def create_blue_to_red_colormap(
    values: list[float], min_val: float | None = None, max_val: float | None = None
) -> list[str]:
    """
    Create a blue to red colormap for given values.

    Args:
        values: List of values to map to colors
        min_val: Minimum value for color mapping (if None, uses min of values)
        max_val: Maximum value for color mapping (if None, uses max of values)

    Returns:
        List of RGB color strings
    """
    if not values:
        return []

    values_array = np.array(values)
    actual_min = min_val if min_val is not None else values_array.min()
    actual_max = max_val if max_val is not None else values_array.max()

    # Avoid division by zero
    if actual_max == actual_min:
        return ["rgb(0, 0, 255)"] * len(values)  # All blue if no variation

    # Normalize values to [0, 1]
    normalized = (values_array - actual_min) / (actual_max - actual_min)

    colors = []
    for norm_val in normalized:
        # Blue to Red gradient
        red = int(255 * norm_val)
        blue = int(255 * (1 - norm_val))
        green = 0
        colors.append(f"rgb({red}, {green}, {blue})")

    return colors


def plot_grid_from_pandapower(
    net: pp.pandapowerNet,
    dig_a_plan: DigAPlan,
    node_size: int = 22,
    width: int = 800,
    height: int = 700,
    from_z: bool = False,
    color_by_results: bool = False,
) -> None:
    """
    Plot grid topology with optional color coding based on voltage and current results.

    Args:
        net: PandaPower network
        dig_a_plan: DigAPlan optimization results
        node_size: Size of node markers
        width: Figure width
        height: Figure height
        from_z: Whether to use z variables for switch status
        color_by_results: If True, color nodes by voltage and edges by current
    """

    # Extract switch status
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

    # Convert PandaPower data to Polars DataFrames
    bus: pl.DataFrame = pl.from_pandas(net["bus"])
    line: pl.DataFrame = pl.from_pandas(net["line"])
    trafo: pl.DataFrame = pl.from_pandas(net["trafo"])
    switch: pl.DataFrame = pl.from_pandas(net["switch"])

    # Configure switch appearance
    switch_mapping = {True: ["1.0", "green", "solid"], False: ["0.3", "red", "dash"]}
    switch = switch.with_columns(
        c("closed")
        .replace_strict(switch_mapping, default=None)
        .list.to_struct(fields=["opacity", "color", "dash"])
        .alias("config"),
    ).unnest("config")

    # Determine node symbols (circle for loads, circle for others)
    load_id = list(net.load["bus"])
    bus = bus.with_columns(
        pl.when(c("bus_id").is_in(load_id))
        .then(pl.lit("circle"))
        .otherwise(pl.lit("circle"))
        .alias("symbol")
    )

    fig = go.Figure()

    # Get voltage and current results if color coding is enabled
    if color_by_results:
        # Extract voltage results
        voltages = dig_a_plan.result_manager.extract_node_voltage()
        voltage_dict = pl_to_dict(voltages.select("node_id", "v_pu"))

        # Extract current results
        currents = dig_a_plan.result_manager.extract_edge_current()
        currents = currents.filter(pl.col("from_node_id") > pl.col("to_node_id"))
        # Create mapping from edge names to current values
        edge_id_mapping = pl_to_dict(
            dig_a_plan.data_manager.edge_data.select("eq_fk", "edge_id")
        )
        current_dict = pl_to_dict(currents.select("edge_id", "i_pct"))

    # Plot lines with current-based coloring if enabled
    if color_by_results and "current_dict" in locals():
        # Get current values for lines
        line_currents = []
        line_names = []
        for _, row in line.to_pandas().iterrows():
            line_name = row["name"]
            line_names.append(line_name)
            if line_name in edge_id_mapping:
                edge_id = edge_id_mapping[line_name]
                current_val = current_dict.get(edge_id, 0.0)
            else:
                current_val = 0.0
            line_currents.append(current_val)

        # Create colors based on current values
        if line_currents:
            line_colors = create_blue_to_red_colormap(line_currents)

            # Plot lines with color mapping
            for i, data in enumerate(line.to_dicts()):
                line_width = 3 if data.get("max_i_ka", 0) > 5e-2 else 2
                color = line_colors[i] if i < len(line_colors) else "blue"

                x_coords = data.get("x_coords", [])
                y_coords = data.get("y_coords", [])

                # Skip if coordinates are empty or invalid
                if not x_coords or not y_coords or len(x_coords) != len(y_coords):
                    continue

                hover_text = (
                    f"Line: {data.get('name', 'N/A')}<br>"
                    f"Current: {line_currents[i]:.3f} pu<br>"
                    f"Max Current: {data.get('max_i_ka', 0):.3f} kA<br>"
                    f"Length: {data.get('length_km', 0):.3f} km"
                )

                # Add visible colored line
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

                # Add hover marker at midpoint
                if len(x_coords) >= 2 and len(y_coords) >= 2:
                    mid_x = (x_coords[0] + x_coords[-1]) / 2
                    mid_y = (y_coords[0] + y_coords[-1]) / 2
                    fig.add_trace(
                        go.Scatter(
                            x=[mid_x],
                            y=[mid_y],
                            mode="markers",
                            marker=dict(
                                size=12, color="rgba(0,0,0,0)"
                            ),  # Invisible marker
                            hoverinfo="text",
                            hovertext=hover_text,
                            showlegend=False,
                            name=f"Line {data.get('name', 'N/A')}",
                        )
                    )
    else:
        # Default line coloring based on capacity
        for data in line.filter(c("max_i_ka") > 5e-2).to_dicts():
            x_coords = data.get("x_coords", [])
            y_coords = data.get("y_coords", [])

            # Skip if coordinates are empty or invalid
            if not x_coords or not y_coords or len(x_coords) != len(y_coords):
                continue

            hover_text = (
                f"Line: {data.get('name', 'N/A')}<br>"
                f"Max Current: {data.get('max_i_ka', 0):.3f} kA<br>"
                f"Length: {data.get('length_km', 0):.3f} km<br>"
                f"Type: High Capacity"
            )

            # Add visible line
            fig.add_trace(
                go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode="lines",
                    line=dict(width=3, color="blue"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

            # Add hover marker at midpoint
            if len(x_coords) >= 2 and len(y_coords) >= 2:
                mid_x = (x_coords[0] + x_coords[-1]) / 2
                mid_y = (y_coords[0] + y_coords[-1]) / 2
                fig.add_trace(
                    go.Scatter(
                        x=[mid_x],
                        y=[mid_y],
                        mode="markers",
                        marker=dict(size=12, color="rgba(0,0,0,0)"),  # Invisible marker
                        hoverinfo="text",
                        hovertext=hover_text,
                        showlegend=False,
                    )
                )

        for data in line.filter(c("max_i_ka") <= 5e-2).to_dicts():
            x_coords = data.get("x_coords", [])
            y_coords = data.get("y_coords", [])

            # Skip if coordinates are empty or invalid
            if not x_coords or not y_coords or len(x_coords) != len(y_coords):
                continue

            hover_text = (
                f"Line: {data.get('name', 'N/A')}<br>"
                f"Max Current: {data.get('max_i_ka', 0):.3f} kA<br>"
                f"Length: {data.get('length_km', 0):.3f} km<br>"
                f"Type: Low Capacity"
            )

            # Add visible line
            fig.add_trace(
                go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode="lines",
                    line=dict(width=3, color="darkviolet"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

            # Add hover marker at midpoint
            if len(x_coords) >= 2 and len(y_coords) >= 2:
                mid_x = (x_coords[0] + x_coords[-1]) / 2
                mid_y = (y_coords[0] + y_coords[-1]) / 2
                fig.add_trace(
                    go.Scatter(
                        x=[mid_x],
                        y=[mid_y],
                        mode="markers",
                        marker=dict(size=12, color="rgba(0,0,0,0)"),  # Invisible marker
                        hoverinfo="text",
                        hovertext=hover_text,
                        showlegend=False,
                    )
                )

    # Plot transformers
    for data in trafo.to_dicts():
        x_coords = data.get("x_coords", [])
        y_coords = data.get("y_coords", [])

        # Skip if coordinates are empty or invalid
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

        # Add visible transformer line
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

        # Add transformer symbol at midpoint
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

    # Plot switches
    for data in switch.to_dicts():
        x_coords = data.get("x_coords", [])
        y_coords = data.get("y_coords", [])

        # Skip if coordinates are empty or invalid
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

        # Add visible switch line (always black, solid for closed, dashed for open)
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

        # Add status indicator box at midpoint
        if len(x_coords) >= 2 and len(y_coords) >= 2:
            mid_x = (x_coords[0] + x_coords[-1]) / 2
            mid_y = (y_coords[0] + y_coords[-1]) / 2

            # Black box for closed, white box for open
            box_color = "black" if is_closed else "white"
            box_line_color = "black"  # Always black border

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

    # Plot buses with voltage-based coloring if enabled
    if color_by_results and "voltage_dict" in locals():
        # Get voltage values for buses
        bus_voltages = []
        bus_ids = bus["bus_id"].to_list()
        for bus_id in bus_ids:
            voltage_val = voltage_dict.get(bus_id, 1.0)
            bus_voltages.append(voltage_val)

        # Create colors based on voltage values
        bus_colors = create_blue_to_red_colormap(
            bus_voltages, min_val=0.95, max_val=1.05  # Typical voltage range
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
                textfont=dict(color="white"),
                marker=dict(
                    size=node_size,
                    color=bus_colors,
                    symbol=bus["symbol"].to_list(),
                    line=dict(color="black", width=1),
                ),
            )
        )
    else:
        # Default bus coloring
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
                marker=dict(
                    size=node_size, color="blue", symbol=bus["symbol"].to_list()
                ),
            )
        )

    # Add load symbols for buses with loads
    for bus_id in load_id:
        # Find the bus coordinates
        bus_row = bus.filter(pl.col("bus_id") == bus_id)
        if len(bus_row) > 0:
            bus_coords = bus_row.select("coords").to_series()[0]
            x_coord = bus_coords[0]
            y_coord = bus_coords[1]

            # Add load symbol slightly offset from the bus
            fig.add_trace(
                go.Scatter(
                    x=[x_coord],
                    y=[y_coord - 0.3],  # Offset below the bus
                    mode="text",
                    text=["⏃"],
                    textfont=dict(size=14, color="orange", family="Arial Black"),
                    hoverinfo="text",
                    hovertext=f"Load at Bus {bus_id}",
                    showlegend=False,
                )
            )

    # Add color scale annotations if color coding is enabled
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
