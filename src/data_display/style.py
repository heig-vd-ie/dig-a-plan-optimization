from plotly.graph_objs import Figure


def apply_plot_style(fig: Figure, x_title: str, y_title: str, title: str) -> None:
    """
    Apply a consistent styling to Plotly figures used in the project.
    """
    fig.update_layout(
        title=dict(
            text=title,
            x=0.55,
            xanchor="center",
            y=0.98,        
            yanchor="top",
            font=dict(family="Times New Roman, Serif", size=22)
        ),
        xaxis_title=x_title,
        yaxis_title=y_title,
        width=1200,
        height=600,
        font=dict(family="Times New Roman, Serif", size=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            title_text="",
            font=dict(family="Times New Roman, Serif", size=18),
            bgcolor="rgba(0,0,0,0)",
            bordercolor="black",
            borderwidth=1,
        ),
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=80, r=40, t=80, b=60),
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="lightgray",
        zeroline=False,
        linecolor="black",
        linewidth=2,
        ticks="outside",
        tickfont=dict(family="Times New Roman, Serif", size=18),
        title_font=dict(family="Times New Roman, Serif", size=20),
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="lightgray",
        zeroline=False,
        linecolor="black",
        linewidth=2,
        ticks="outside",
        tickfont=dict(family="Times New Roman, Serif", size=18),
        title_font=dict(family="Times New Roman, Serif", size=20),
    )
