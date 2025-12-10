import os
from typing import Dict, Literal
import polars as pl
import plotly.express as px
from pipelines.reconfiguration import DigAPlanADMM
from data_display.style import apply_plot_style



class DistributionVariable:
    def __init__(
        self,
        daps: Dict[str, DigAPlanADMM],
        variable_name: str,
        variable_type: Literal["nodal", "edge"] = "nodal",
    ):
        self.daps = daps
        self.variable_name = variable_name
        self.variable_type = variable_type

    def extract_variable_distribution(
        self,
        dap: DigAPlanADMM,
    ) -> pl.DataFrame:
        all_data = []
        for ω in range(len(dap.model_manager.admm_model_instances)):
            match self.variable_name:
                case "voltage":
                    data = dap.result_manager.extract_node_voltage(ω)
                case "current":
                    data = dap.result_manager.extract_edge_current(ω)
                case _:
                    if self.variable_type == "nodal":
                        data = dap.result_manager.extract_nodal_variables(
                            self.variable_name, ω
                        )
                    elif self.variable_type == "edge":
                        data = dap.result_manager.extract_edge_variables(
                            self.variable_name, ω
                        )
            data = data.with_columns(pl.lit(ω).alias("scenario"))
            all_data.append(data)
        return pl.concat(all_data)

    def merge_variables(self) -> pl.DataFrame:
        merged_data = []
        for dap_name, dap in self.daps.items():
            data = self.extract_variable_distribution(dap)
            data = data.with_columns(pl.lit(dap_name).alias("method"))
            merged_data.append(data)
        return pl.concat(merged_data)

    def variable_name_alias(self) -> str:
        if self.variable_name == "voltage":
            return "v_pu"
        if self.variable_name == "current":
            return "i_pct"
        return self.variable_name

    def voltage_min_limit(self):
        return (
            self.daps[next(iter(self.daps))].data_manager.node_data["v_min_pu"].mean()
        )

    def voltage_max_limit(self):
        return (
            self.daps[next(iter(self.daps))].data_manager.node_data["v_max_pu"].mean()
        )

    def plot_distribution_variable(self) -> None:
        df = self.merge_variables()
        if self.variable_type == "edge":
            df = df.filter(pl.col("from_node_id") > pl.col("to_node_id"))
        df_pd = df.to_pandas()
        labels = {
            "node_id": "Bus ID",
            "edge_id": "Edge ID",
            self.variable_name_alias(): f"{self.variable_name.capitalize()} (p.u.)",
            "method": "Method",
        }
        fig = px.box(
            df_pd,
            x="node_id" if self.variable_type == "nodal" else "edge_id",
            y=self.variable_name_alias(),
            color="method",
            title=f"{self.variable_name.capitalize()} Distribution by Bus",
            labels=labels,
            hover_data=["scenario"],
            points=False,
        )

        if self.variable_name == "voltage":
            fig.add_hline(
                y=self.voltage_min_limit(),
                line_dash="dash",
                line_color="red",
                annotation_text="Min Voltage Limit",
            )
            fig.add_hline(
                y=self.voltage_max_limit(),
                line_dash="dash",
                line_color="red",
                annotation_text="Max Voltage Limit",
            )
            fig.add_hline(
                y=1.0,
                line_dash="solid",
                line_color="green",
                annotation_text="Nominal Voltage",
            )
        elif self.variable_name == "current":
            fig.add_hline(
                y=100.0,
                line_dash="solid",
                line_color="red",
                annotation_text="Max Current limit",
            )
        
        # === Apply shared style helper ===
        x_title = "Bus ID" if self.variable_type == "nodal" else "Edge ID"
        y_unit = "%" if self.variable_name == "current" else "p.u."
        y_title = f"{self.variable_name.capitalize()} ({y_unit})"
        title = f"{self.variable_name.capitalize()} Distribution by Bus"
        
        apply_plot_style(
            fig,
            x_title=x_title,
            y_title=y_title,
            title=title,
        )

        fig.update_xaxes(categoryorder="category ascending")
        os.makedirs(".cache/figs", exist_ok=True)
        fig.write_html(f".cache/figs/distribution_{self.variable_name}.html")


def plot_distribution_variable(
    daps: Dict[str, DigAPlanADMM],
    variable_name: str,
    variable_type: Literal["nodal", "edge"] = "nodal",
) -> None:
    distribution = DistributionVariable(daps, variable_name, variable_type)
    distribution.plot_distribution_variable()
