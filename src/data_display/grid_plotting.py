from plotly.graph_objects import Figure
import plotly.graph_objs as go
import pandapower as pp
import polars as pl
from polars import col as c

def plot_grid_from_pandapower(net: pp.pandapowerNet, node_size: int = 22, width: int=800,  height: int=700) -> None:
    
    bus: pl.DataFrame = pl.from_pandas(net["bus"])
    line: pl.DataFrame = pl.from_pandas(net["line"])
    trafo: pl.DataFrame = pl.from_pandas(net["trafo"])
    switch: pl.DataFrame = pl.from_pandas(net["switch"])
    switch = switch.with_columns(
        c("closed").replace_strict({True:1.0,False:0.3},default=None).alias("opacity")
    )


    fig = go.Figure()

    for data in line.to_dicts():
            fig.add_trace(
                go.Scatter(
                    x=data["x_coords"],
                    y=data["y_coords"],
                    mode="lines",
                    line=dict(width=3, color="blue"),
                    hoverinfo="none",
                    showlegend=False
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
                    showlegend=False
                )
            )

    # for data in switch.to_dicts():
    #         fig.add_trace(
    #             go.Scatter(
    #                 x=data["x_coords"],
    #                 y=data["y_coords"],
    #                 mode="lines",
    #                 opacity= data["opacity"],
    #                 line=dict(width=3, color="darkred"),
    #                 hoverinfo="none",
    #                 showlegend=False
    #             )
    #         )
    
    
    
    # draw switchable lines: closed = green solid
    for data in switch.filter(c("closed") == True).to_dicts():
        fig.add_trace(go.Scatter(
            x=data["x_coords"],
            y=data["y_coords"],
            mode="lines",
            opacity=data["opacity"],
            line=dict(width=3, color="green", dash="solid"),
            hoverinfo="none",
            showlegend=False,
        ))

    # draw switchable lines: open = red dashed
    for data in switch.filter(c("closed") == False).to_dicts():
        fig.add_trace(go.Scatter(
            x=data["x_coords"],
            y=data["y_coords"],
            mode="lines",
            opacity=data["opacity"],
            line=dict(width=3, color="red", dash="dash"),
            hoverinfo="none",
            showlegend=False,
        ))
    
            
    fig.add_trace(
            go.Scatter(
                x=bus.select(c("coords").list.get(0))["coords"].to_list(),
                y=bus.select(c("coords").list.get(1))["coords"].to_list(),
                text=bus.with_columns("</b>" + c("bus_id").cast(pl.Utf8) + "</b>" )["bus_id"].to_list(),
                mode="markers+text",
                hoverinfo="none",
                showlegend=False,
                textfont=dict(color="white"),
                marker=dict(
                    size=node_size, color="blue"
                ),
            )
        )

    fig.update_layout(
        margin=dict(t=5, l=65, r= 10, b=5), 
        width=width,   # Set the width of the figure
        height=height,
        paper_bgcolor='white', plot_bgcolor='white',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )
    fig.show()