import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os


class MyPlotter:
    def __init__(self):

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
        df: pd.DataFrame,
        field: str,
        field_name: str,
        save_name: str | None = None,
        nbins: int = 50,
    ):
        # Create high-resolution histogram
        filtered_df = df[
            (df["risk_method"] != "Expectation")
            | (df["iteration"] == df["iteration"].max())
        ]
        filtered_df = filtered_df[
            (filtered_df["risk_method"] != "Wasserstein")
            | (filtered_df["risk_param"] == 0.1)
        ]
        fig = px.histogram(
            filtered_df,
            x=field,
            color="risk_label",
            nbins=nbins,  # Reduced from 100 to make bars bigger
            histnorm="probability density",
            title="",
            labels={
                field: field_name,
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

    def create_box_plot(
        self,
        df: pd.DataFrame,
        field: str,
        field_name: str,
        save_name: str | None = None,
    ):
        fig = px.box(
            df[
                (df["risk_method"] != "Expectation")
                | (df["iteration"] == df["iteration"].max())
            ],
            x="risk_label",
            color="risk_label",
            y=field,
            title="",
            labels={
                field: field_name,
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

    def create_parallel_coordinates_plot(
        self,
        df: pd.DataFrame,
        field: str,
        risk_label: str,
        value_col: str,
        stage_col: str = "stage",
        save_name: str | None = None,
        title: str = "",
    ):
        """
        Create a parallel coordinates plot for evolution over stages.

        Each line represents one simulation (risk_label) showing the average
        values across all stages. Perfect for visualizing how different
        simulations evolve over time periods.

        Parameters:
        -----------
        risk_label : str
            Specific risk label to filter and plot
        stage_col : str, optional (default: "stage")
            Column name for stages/time periods
        value_col : str, optional (default: None)
            Column name for values to plot. If None, uses self.field
        save_name : str, optional (default: None)
            Name for saving the figure
        title : str, optional (default: "")
            Title for the plot
        show_target_increase : bool, optional (default: False)
            Whether to show a target percentage increase line
        target_increase_percent : float, optional (default: 2.0)
            Target percentage increase per stage (e.g., 2.0 for 2% increase)
        """

        filtered_df = df[(df["risk_label"] == risk_label)]

        # Check if required columns exist
        required_cols = [stage_col, value_col]
        missing_cols = [col for col in required_cols if col not in filtered_df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        # Handle duplicates by aggregating (taking mean) - this ensures each
        # simulation has one average value per stage
        df_agg = (
            filtered_df.groupby([stage_col, "simulation"])[value_col]
            .mean()
            .reset_index()
        )

        # Pivot the data to have stages as columns
        # Each row = one simulation, each column = one stage
        df_pivot = df_agg.pivot(index="simulation", columns=stage_col, values=value_col)

        # Reset index to make risk_label a column
        df_pivot = df_pivot.reset_index()

        # Get stage columns (assuming they are numeric or can be sorted)
        stage_columns = [col for col in df_pivot.columns if col != "simulation"]
        try:
            stage_columns = sorted(stage_columns, key=lambda x: float(x) + 1)
        except (ValueError, TypeError):
            stage_columns = sorted(stage_columns)

        # Handle missing stage columns gracefully
        if not stage_columns:
            raise ValueError(
                f"No stage columns found. Available columns: {df_pivot.columns.tolist()}"
            )

        # Create dimension list for parallel coordinates
        # Each dimension = one vertical axis in the plot

        # Calculate global min and max across all stages for consistent scaling
        all_values = []
        valid_stage_columns = []
        for col in stage_columns:
            # Skip columns with all NaN values
            if df_pivot[col].notna().sum() == 0:
                continue
            valid_stage_columns.append(col)
            all_values.extend(df_pivot[col].dropna().tolist())

        if not all_values:
            raise ValueError("No valid values found for parallel coordinates plot")

        # Use the same range for all stages
        global_min = min(all_values)
        global_max = max(all_values)

        # Add some padding to make the plot look better (optional)
        value_range = global_max - global_min
        padding = value_range * 0.05  # 5% padding
        global_min -= padding
        global_max += padding

        dimensions = []
        for col in valid_stage_columns:
            dimensions.append(
                dict(
                    label=f"Year {col * 5}",
                    values=df_pivot[col],
                    range=[global_min, global_max],  # Same range for all stages
                )
            )

        if not dimensions:
            raise ValueError("No valid dimensions found for parallel coordinates plot")

        # Create risk_label mapping for coloring - each simulation gets a different color
        if df_pivot["simulation"].dtype == "object":
            risk_label_map = {
                label: i for i, label in enumerate(df_pivot["simulation"].unique())
            }
            df_pivot["risk_label_numeric"] = df_pivot["simulation"].map(risk_label_map)
            color_col = "risk_label_numeric"
        else:
            color_col = "simulation"

        # Create the parallel coordinates plot
        # Each line connects the values for one simulation across all stages
        fig = go.Figure(
            data=go.Parcoords(
                line=dict(
                    color=df_pivot[color_col],
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(
                        title="Simulation",
                        title_font=dict(size=self.font_size, family=self.font_family),
                        tickfont=dict(size=self.font_size, family=self.font_family),
                    ),
                ),
                dimensions=dimensions,
                labelangle=45,
                labelside="bottom",
            )
        )

        # Apply scientific formatting (adapted for parallel coordinates)
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
            margin=dict(l=60, r=60, t=60, b=60),  # More margin for parallel coords
        )

        # Save if name provided
        if save_name:
            self.save_figure(fig, save_name)

        return fig
