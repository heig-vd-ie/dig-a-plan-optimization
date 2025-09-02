import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_histogram_plot(df, config):
    fig = px.histogram(
        df,
        x="objective",
        color="risk_label",
        nbins=config.histogram_bins,
        histnorm="probability density",
        title="Distribution of Objectives by Risk Method",
        labels={
            "objective": "Objective value",
            "count": "Density",
            "risk_label": "Risk Method",
        },
        barmode="group",
    )

    fig.update_layout(
        width=config.plot_width,
        height=config.plot_height,
        legend=dict(title="Risk Method", orientation="v", x=1.02, y=1),
        margin=dict(r=200),
    )

    return fig


def create_box_plot(df, config):
    fig = px.box(
        df,
        x="risk_label",
        y="objective",
        title="Objective Value Distributions by Risk Method",
        labels={
            "objective": "Objective Value",
            "risk_label": "Risk Method",
        },
    )

    fig.update_layout(
        width=config.plot_width, height=config.plot_height, xaxis_tickangle=-45
    )

    return fig


def create_iteration_plot(df, config):
    fig = px.line(
        df,
        x="iteration",
        y="objective",
        color="risk_label",
        title="Objective Value Evolution by Iteration",
        labels={
            "iteration": "Iteration",
            "objective": "Objective Value",
            "risk_label": "Risk Method",
        },
    )

    fig.update_layout(
        width=config.plot_width,
        height=config.plot_height,
        legend=dict(title="Risk Method", orientation="v", x=1.02, y=1),
        margin=dict(r=200),
    )

    return fig


def create_scatter_plot(df, config):
    fig = px.scatter(
        df,
        x="iteration",
        y="objective",
        color="risk_label",
        title="Objective Values by Iteration (Scatter)",
        labels={
            "iteration": "Iteration",
            "objective": "Objective Value",
            "risk_label": "Risk Method",
        },
        opacity=0.7,
    )

    fig.update_layout(
        width=config.plot_width,
        height=config.plot_height,
        legend=dict(title="Risk Method", orientation="v", x=1.02, y=1),
        margin=dict(r=200),
    )

    return fig


def create_comparison_dashboard(df, config):
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=["Histogram", "Box Plot", "Iteration Evolution", "Scatter Plot"],
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
        ],
    )

    return fig
