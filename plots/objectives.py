import pandas as pd
import plotly.express as px
from plots import Config


class MyObjectivePlotter:
    def __init__(self, df: pd.DataFrame, config: Config):
        self.df = df
        self.config = config

    def create_histogram_plot(self):
        fig = px.histogram(
            self.df,
            x="objective",
            color="risk_label",
            nbins=self.config.histogram_bins,
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
            width=self.config.plot_width,
            height=self.config.plot_height,
            legend=dict(title="Risk Method", orientation="v", x=1.02, y=1),
            margin=dict(r=200),
        )

        return fig

    def create_box_plot(self):
        fig = px.box(
            self.df,
            x="risk_label",
            y="objective",
            title="Objective Value Distributions by Risk Method",
            labels={
                "objective": "Objective Value",
                "risk_label": "Risk Method",
            },
        )

        fig.update_layout(
            width=self.config.plot_width,
            height=self.config.plot_height,
            xaxis_tickangle=-45,
        )

        return fig

    def create_iteration_plot(self):
        fig = px.line(
            self.df,
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
            width=self.config.plot_width,
            height=self.config.plot_height,
            legend=dict(title="Risk Method", orientation="v", x=1.02, y=1),
            margin=dict(r=200),
        )

        return fig

    def create_scatter_plot(self):
        fig = px.scatter(
            self.df,
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
            width=self.config.plot_width,
            height=self.config.plot_height,
            legend=dict(title="Risk Method", orientation="v", x=1.02, y=1),
            margin=dict(r=200),
        )

        return fig
