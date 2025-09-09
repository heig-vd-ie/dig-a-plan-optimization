from typing import Dict, List
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os


class MyPlotter:
    def __init__(self):

        # Scientific paper formatting attributes
        self.template = "plotly_white"
        self.font_size = 12  
        self.marker_size = 7  
        self.colors = px.colors.qualitative.Dark24  
        self.width = 600  
        self.height = 400  
        self.grid_color = "lightgray"  
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
                    standoff=10,  
                ),
            ),
            legend=dict(
                font=dict(size=self.font_size, family=self.font_family),
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1,
            ),
            margin=dict(l=40, r=15, t=40, b=40),  
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
        with_respect_to: str = "risk_label",
        save_name: str | None = None,
        nbins: int = 50,
    ):
        # Create high-resolution histogram
        if with_respect_to == "risk_label":
            filtered_df = df[
                (df["risk_method"] != "Expectation")
                | (df["iteration"] == df["iteration"].max())
            ]
            filtered_df = filtered_df[
                (filtered_df["risk_method"] != "Wasserstein")
                | (filtered_df["risk_param"] == 0.1)
            ]
        else:
            filtered_df = df.copy()
        fig = px.histogram(
            filtered_df,
            x=field,
            color=with_respect_to,
            nbins=nbins,  
            histnorm="probability density",
            title="",
            labels={
                field: field_name,
                "count": "Density",
                with_respect_to: "",
            },
            opacity=0.8,  
            barmode="group",  
            color_discrete_sequence=self.colors,  
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
        with_respect_to: str = "risk_label",
        save_name: str | None = None,
    ):
        if with_respect_to == "risk_label":
            filtered_df = df[
                (df["risk_method"] != "Expectation")
                | (df["iteration"] == df["iteration"].max())
            ]
            filtered_df = filtered_df[
                (filtered_df["risk_method"] != "Wasserstein")
                | (filtered_df["risk_param"] == 0.1)
            ]
        else:
            filtered_df = df.copy()
        fig = px.box(
            filtered_df,
            x=with_respect_to,
            color=with_respect_to,
            y=field,
            title="",
            labels={
                field: field_name,
                with_respect_to: with_respect_to.replace("_", " ").title(),
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
        risk_labels: list[str],
        value_col: str = "",
        field_name: str = "",
        stage_col: str = "stage",
        risk_label_col: str = "risk_label",
        simulation_col: str = "simulation",
        save_name: str | None = None,
        title_prefix: str = "",
        # NEW: divide each series by the per-stage mean of this label (e.g., "Expectation (α=0.1)")
        normalize_to_label: str | None = None,
    ) -> Dict[str, go.Figure]:
        """
        Build one parallel-coordinates plot per risk label.

        If `normalize_to_label` is provided, each stage value is divided by the
        per-stage mean of that baseline label (computed over all simulations of that label).
        """
        # --- Basic checks
        required_cols = [stage_col, value_col, risk_label_col, simulation_col]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # Filter to the labels we care about
        filtered_df = df[df[risk_label_col].isin(risk_labels + ([normalize_to_label] if normalize_to_label else []))].copy()

        # --- Compute baseline per-stage means if normalization requested
        baseline_means = None
        if normalize_to_label is not None:
            base_df = filtered_df[filtered_df[risk_label_col] == normalize_to_label]
            if base_df.empty:
                print(f"Warning: no rows for baseline '{normalize_to_label}'. Skipping normalization.")
                normalize_to_label = None
            else:
                base_agg = (
                    base_df.groupby(stage_col, dropna=False)[value_col]
                    .mean()
                    .reset_index()
                )
                baseline_means = dict(zip(base_agg[stage_col], base_agg[value_col]))

        figures: Dict[str, go.Figure] = {}

        for risk_label in risk_labels:
            # Slice this label
            risk_df = filtered_df[filtered_df[risk_label_col] == risk_label]
            if risk_df.empty:
                continue

            # Aggregate duplicates by mean across (stage, simulation)
            df_agg = (
                risk_df.groupby([stage_col, simulation_col], dropna=False)[value_col]
                .mean()
                .reset_index()
            )

            # Pivot so stages become columns
            df_pivot = df_agg.pivot_table(
                index=simulation_col,
                columns=stage_col,
                values=value_col,
                aggfunc="mean",
            )

            # Sort stage columns numerically when possible
            stage_columns = list(df_pivot.columns)
            def _to_float(x):
                try:
                    return float(x)
                except Exception:
                    return None

            sortable = [(_to_float(c), c) for c in stage_columns]
            # Keep numeric stages sorted, then any non-numeric at the end in original order
            numeric_sorted = [c for f, c in sorted((p for p in sortable if p[0] is not None), key=lambda t: t[0])] #type: ignore
            non_numeric = [c for f, c in sortable if f is None]
            stage_columns = numeric_sorted + non_numeric

            # Reorder columns and flatten index
            df_pivot = df_pivot[stage_columns].reset_index()

            # --- Apply normalization (divide by baseline per-stage mean)
            if normalize_to_label is not None and baseline_means is not None:
                for col in stage_columns:
                    base = baseline_means.get(col, None)
                    if base is not None and base != 0:
                        df_pivot[col] = df_pivot[col] / base
                    else:
                        # If no baseline for this stage, leave as NaN so it won't plot
                        df_pivot[col] = float("nan")

            # Keep only stages that have at least one non-NaN value
            valid_stage_cols = [c for c in stage_columns if df_pivot[c].notna().any()]
            if not valid_stage_cols:
                continue

            # Compute global range with padding; handle constant case
            values_series = pd.concat([df_pivot[c].dropna() for c in valid_stage_cols])
            if values_series.empty:
                continue
            global_min = float(values_series.min()) #type: ignore
            global_max = float(values_series.max()) #type: ignore
            if global_max == global_min:
                pad = abs(global_min) * 0.05 if global_min != 0 else 0.05
                global_min -= pad
                global_max += pad
            else:
                pad = (global_max - global_min) * 0.05
                global_min -= pad
                global_max += pad

            # Build dimensions
            dimensions = []
            for col in valid_stage_cols:
                # If stage is numeric, label as Year {5 * stage}; else use raw label
                try:
                    lbl = f"Year {int(round(float(col) * 5))}"
                except Exception:
                    lbl = str(col)
                dimensions.append(
                    dict(
                        label=lbl,
                        values=df_pivot[col],
                        range=[global_min, global_max],
                    )
                )

            # Color lines by (encoded) simulation id
            unique_sims = sorted(df_pivot[simulation_col].astype(str).unique())
            sim_map = {sim: i for i, sim in enumerate(unique_sims)}
            color_vals = df_pivot[simulation_col].astype(str).map(sim_map)

            fig = go.Figure(
                data=go.Parcoords(
                    line=dict(
                        color=color_vals,
                        colorscale="Viridis",
                        showscale=False,
                    ),
                    dimensions=dimensions,
                    labelangle=45,
                    labelside="bottom",
                )
            )

            # Layout and title
            plot_title = f"{title_prefix}{risk_label} - {field_name}"
            if normalize_to_label:
                plot_title += f" / mean({normalize_to_label})"
            fig.update_layout(
                template=self.template,
                width=self.width + 100,  # room for labels
                height=self.height,
                title=dict(
                    text=plot_title,
                    font=dict(size=self.font_size + 2, family=self.font_family),
                    x=0.5,
                    xanchor="center",
                ),
                font=dict(size=self.font_size, family=self.font_family),
                plot_bgcolor="white",
                paper_bgcolor="white",
                margin=dict(l=60, r=120, t=60, b=60),
            )

            figures[risk_label] = fig

            # Save each figure if requested
            if save_name:
                safe_risk_label = (
                    risk_label.replace(" ", "_")
                    .replace("(", "")
                    .replace(")", "")
                    .replace("α=", "alpha")
                )
                suffix = "_ratio" if normalize_to_label else ""
                self.save_figure(fig, f"{save_name}{suffix}_{safe_risk_label}")

        return figures



