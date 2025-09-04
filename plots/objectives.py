import pandas as pd
import plotly.express as px
from .mongo_client import MongoConfig


class MyObjectivePlotter:
    def __init__(self, df: pd.DataFrame, config: MongoConfig, field: str):
        self.df = df
        self.config = config
        self.field = field

    def create_histogram_plot(self):
        fig = px.histogram(
            self.df[
                (self.df["risk_method"] != "Expectation")
                | (self.df["iteration"] == self.df["iteration"].max())
            ],
            x=self.field,
            color="risk_label",
            nbins=self.config.histogram_bins,
            histnorm="probability density",
            title="Distribution of Objectives by Risk Method",
            labels={
                self.field: "Objective value",
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
            self.df[
                (self.df["risk_method"] != "Expectation")
                | (self.df["iteration"] == self.df["iteration"].max())
            ],
            x="risk_label",
            y=self.field,
            title="Objective Value Distributions by Risk Method",
            labels={
                self.field: "Objective Value",
                "risk_label": "Risk Method",
            },
        )

        fig.update_layout(
            width=self.config.plot_width,
            height=self.config.plot_height,
            xaxis_tickangle=-45,
        )

        return fig

    def create_scatter_plot(self):
        fig = px.box(
            self.df[self.df["risk_method"] == "Expectation"],
            x="iteration",
            y=self.field,
            color="risk_label",
            title="Objective Values by Iteration",
            labels={
                "iteration": "Iteration",
                self.field: "Objective Value",
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
