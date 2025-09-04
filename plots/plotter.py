import pandas as pd
import plotly.express as px


class MyPlotter:
    def __init__(self, df: pd.DataFrame, field: str):
        self.df = df
        self.field = field

    def create_histogram_plot(self, field_name: str):
        # Create high-resolution histogram
        fig = px.histogram(
            self.df[
                (self.df["risk_method"] != "Expectation")
                | (self.df["iteration"] == self.df["iteration"].max())
            ],
            x=self.field,
            color="risk_label",
            nbins=100,  # Much higher resolution
            histnorm="probability density",
            title="",
            labels={
                self.field: field_name,
                "count": "Density",
                "risk_label": "",
            },
            opacity=0.8,  # Slightly higher opacity since they're side by side
            barmode="group",  # Side by side bars for clear comparison
        )

        # Make bars look clean with minimal gaps
        fig.update_traces(
            marker=dict(
                line=dict(
                    width=0.2, color="rgba(255,255,255,0.5)"
                )  # Thin white borders
            )
        )

        fig.update_layout(
            width=800,
            height=600,
            # Publication-ready legend
            legend=dict(
                orientation="h",
                x=0.5,
                y=0.98,
                xanchor="center",
                yanchor="top",
                title="",
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1,
                font=dict(size=14, family="Arial, sans-serif"),
            ),
            # Adjusted margins for larger fonts
            margin=dict(l=60, r=20, t=50, b=60),
            # Publication-ready y-axis with grid
            yaxis=dict(
                title=dict(
                    text="Density", font=dict(size=16, family="Arial, sans-serif")
                ),
                showgrid=True,
                gridwidth=2,  # Slightly thicker grid lines
                gridcolor="rgba(128,128,128,0.4)",  # More visible grid
                showline=True,
                linewidth=2,
                linecolor="Black",
                tickmode="linear",
                tick0=0,  # Start ticks from 0
                dtick=None,  # Let Plotly auto-determine tick spacing
                nticks=8,  # Suggest number of ticks
                showticklabels=True,  # Ensure tick labels are shown
                tickfont=dict(size=14, family="Arial, sans-serif"),
                mirror=True,
                zeroline=True,  # Show zero line
                zerolinewidth=2,
                zerolinecolor="rgba(128,128,128,0.6)",
            ),
            # Publication-ready x-axis with grid
            xaxis=dict(
                title=dict(
                    text=field_name, font=dict(size=16, family="Arial, sans-serif")
                ),
                showgrid=True,
                gridwidth=2,  # Slightly thicker grid lines
                gridcolor="rgba(128,128,128,0.4)",  # More visible grid
                showline=True,
                linewidth=2,
                linecolor="Black",
                tickfont=dict(size=14, family="Arial, sans-serif"),
                mirror=True,
                zeroline=True,  # Show zero line
                zerolinewidth=2,
                zerolinecolor="rgba(128,128,128,0.6)",
            ),
            # Clean publication background
            plot_bgcolor="white",
            paper_bgcolor="white",
            # Show legend
            showlegend=True,
            # Adjust bar gap for better appearance
            bargap=0.1,  # Small gap between groups
            bargroupgap=0.05,  # Very small gap between bars in a group
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
            width=800,
            height=600,
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
            width=800,
            height=600,
            legend=dict(title="Risk Method", orientation="v", x=1.02, y=1),
            margin=dict(r=200),
        )

        return fig
