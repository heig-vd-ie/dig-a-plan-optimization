import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import os


class MyPlotter:
    def __init__(self, df: pd.DataFrame, field: str):
        self.df = df
        self.field = field

        # Scientific paper formatting attributes
        self.template = "plotly_white"
        self.font_size = 12  # ≥ 10–12 pt for journals
        self.marker_size = 7  # 6–8 px for clarity
        self.colors = px.colors.qualitative.Dark24  # Colorblind-friendly palette
        self.width = 600  # Single column width
        self.height = 400  # Maintain good aspect ratio
        self.grid_color = "lightgray"  # Light gray gridlines
        self.font_family = "Arial, sans-serif"
        self.cache_dir = ".cache/figs"

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

    def set_formatting(self, **kwargs):
        """
        Customize formatting attributes for scientific papers.

        Parameters:
        -----------
        font_size : int, optional (default: 12)
            Font size in points (≥ 10–12 pt for journals)
        marker_size : int, optional (default: 7)
            Marker size in pixels (6–8 px for clarity)
        width : int, optional (default: 600)
            Figure width in pixels (single column ~8.5 cm)
        height : int, optional (default: 400)
            Figure height in pixels
        grid_color : str, optional (default: "lightgray")
            Grid line color
        font_family : str, optional (default: "Arial, sans-serif")
            Font family
        colors : list, optional (default: px.colors.qualitative.Safe)
            Color palette (colorblind-friendly recommended)
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: {key} is not a valid formatting attribute")

    def _apply_scientific_formatting(self, fig, title=""):
        """Apply scientific paper formatting to a plotly figure."""
        fig.update_layout(
            template=self.template,
            width=self.width,
            height=self.height,
            title=dict(
                text=title,
                font=dict(size=self.font_size + 2, family=self.font_family),
                x=0.5,
                xanchor="center",
            ),
            font=dict(size=self.font_size, family=self.font_family),
            plot_bgcolor="white",
            paper_bgcolor="white",
            # Grid styling
            xaxis=dict(
                showgrid=True,
                gridcolor=self.grid_color,
                gridwidth=1,
                showline=True,
                linewidth=1,
                linecolor="black",
                mirror=True,
                tickfont=dict(size=self.font_size, family=self.font_family),
                title=dict(
                    font=dict(size=self.font_size + 1, family=self.font_family),
                    standoff=10,
                ),
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor=self.grid_color,
                gridwidth=1,
                showline=True,
                linewidth=1,
                linecolor="black",
                mirror=True,
                tickfont=dict(size=self.font_size, family=self.font_family),
                title=dict(
                    font=dict(size=self.font_size + 1, family=self.font_family),
                    standoff=10,  # Move y-axis label closer
                ),
            ),
            legend=dict(
                font=dict(size=self.font_size, family=self.font_family),
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1,
            ),
            margin=dict(l=40, r=15, t=40, b=40),  # Reduced margins significantly
        )

        return fig

    def save_figure(self, fig, name):
        """Save figure to cache directory in SVG format."""
        filepath = os.path.join(self.cache_dir, f"{name}.svg")
        try:
            fig.write_image(filepath, format="svg")
            html_filepath = os.path.join(self.cache_dir, f"{name}.html")
            fig.write_html(html_filepath)
        except Exception as e:
            print(f"Warning: Could not save SVG format. Error: {e}")
            print("To enable SVG export, install kaleido: pip install kaleido")
            # Fallback to HTML
            html_filepath = os.path.join(self.cache_dir, f"{name}.html")
            fig.write_html(html_filepath)
            return html_filepath
        return filepath

    def create_histogram_plot(
        self,
        field_name: str,
        save_name: str | None = None,
        nbins: int = 50,
    ):
        # Create high-resolution histogram
        filtered_df = self.df[
            (self.df["risk_method"] != "Expectation")
            | (self.df["iteration"] == self.df["iteration"].max())
        ]
        filtered_df = filtered_df[
            (filtered_df["risk_method"] != "Wasserstein")
            | (filtered_df["risk_param"] == 0.1)
        ]
        fig = px.histogram(
            filtered_df,
            x=self.field,
            color="risk_label",
            nbins=nbins,  # Reduced from 100 to make bars bigger
            histnorm="probability density",
            title="",
            labels={
                self.field: field_name,
                "count": "Density",
                "risk_label": "",
            },
            opacity=0.8,  # Slightly higher opacity since they're side by side
            barmode="group",  # Side by side bars for clear comparison
            color_discrete_sequence=self.colors,  # Use colorblind-friendly palette
        )

        # Make bars look clean with minimal gaps
        fig.update_traces(
            marker=dict(
                line=dict(
                    width=0.2, color="rgba(255,255,255,0.5)"
                )  # Thin white borders
            )
        )

        # Apply scientific formatting
        fig = self._apply_scientific_formatting(fig)

        # Specific updates for histogram
        fig.update_layout(
            # Publication-ready legend
            legend=dict(
                orientation="h",
                x=0.5,
                y=0.98,
                xanchor="center",
                yanchor="top",
                title="",
            ),
            # Adjust bar gap for better appearance - smaller gaps for bigger bars
            bargap=0.05,  # Reduced gap between groups
            bargroupgap=0.02,  # Very small gap between bars in a group
            # Update axis labels
            xaxis_title=field_name,
            yaxis_title="Density",
        )

        # Save if name provided
        if save_name:
            self.save_figure(fig, save_name)

        return fig

    def create_box_plot(self, field_name: str, save_name: str | None = None):
        fig = px.box(
            self.df[
                (self.df["risk_method"] != "Expectation")
                | (self.df["iteration"] == self.df["iteration"].max())
            ],
            x="risk_label",
            color="risk_label",
            y=self.field,
            title="",
            labels={
                self.field: field_name,
                "risk_label": "Risk Method",
            },
            color_discrete_sequence=self.colors,  # Use colorblind-friendly palette
        )

        # Apply scientific formatting
        fig = self._apply_scientific_formatting(fig)

        # Specific updates for box plot
        fig.update_layout(
            yaxis_title=field_name,
            showlegend=False,  # Turn off legend
            xaxis_tickangle=-20,
        )

        # Save if name provided
        if save_name:
            self.save_figure(fig, save_name)

        return fig
