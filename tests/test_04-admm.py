import pytest
import polars as pl
import math
from data_display.output_processing import compare_dig_a_plan_with_pandapower
from data_exporter.pandapower_to_dig_a_plan import pandapower_to_dig_a_plan_schema
from pipelines.reconfiguration import DigAPlanADMM, DigAPlanCombined
from pipelines.reconfiguration.configs import ADMMConfig, CombinedConfig, PipelineType


class TestADMMModel:
    @pytest.fixture(autouse=True)
    def setup_common_data(
        self,
        test_simple_grid,
        test_taps,
        test_admm_config,
        test_combined_config,
        test_simple_grid_groups,
    ):
        """Set up common test data and configurations."""
        self.net = test_simple_grid
        self.taps = test_taps
        self.admm_config: ADMMConfig = test_admm_config
        self.combined_config = test_combined_config
        self.simple_grid_groups = test_simple_grid_groups


class TestADMMModelSimpleExample(TestADMMModel):
    def test_admm_model_simple_example(self):

        grid_data = pandapower_to_dig_a_plan_schema(self.net, taps=[100])

        config = self.admm_config
        config.groups = self.simple_grid_groups

        dap = DigAPlanADMM(config=config)

        dap.add_grid_data(grid_data)

        dap.model_manager.solve_model()

        switches1 = dap.data_manager.edge_data.filter(
            pl.col("type") == "switch"
        ).select("eq_fk", "edge_id", "normal_open")

        z_df = pl.DataFrame(
            {
                "edge_id": list(dap.model_manager.zδ.keys()),
                "z": list(dap.model_manager.zδ.values()),
            }
        )

        consensus_states = (
            switches1.join(z_df, on="edge_id", how="inner")
            .with_columns(
                (pl.col("z") > 0.5).alias("closed"),
                (~(pl.col("z") > 0.5)).alias("open"),
            )
            .select("eq_fk", "edge_id", "z", "normal_open", "closed", "open")
            .sort("edge_id")
        )

        node_data, edge_data = compare_dig_a_plan_with_pandapower(
            dig_a_plan=dap, net=self.net
        )
        assert node_data.get_column("v_diff").abs().max() < 1e-6  # type: ignore
        assert math.isclose(edge_data.get_column("i_diff").abs().max(), 0.0009804, rel_tol=1e-3)  # type: ignore

        config = CombinedConfig(
            verbose=True,
            big_m=1e3,
            ε=1,
            pipeline_type=PipelineType.COMBINED,
            γ_infeasibility=1.0,
            γ_admm_penalty=0.0,
            all_scenarios=True,
        )

        dig_a_plan = DigAPlanCombined(config=config)

        dig_a_plan.add_grid_data(grid_data)
        dig_a_plan.solve_model()  # one‐shot solve

        # Switch status
        switches2 = dig_a_plan.result_manager.extract_switch_status()

        assert (
            consensus_states.select(["eq_fk", "open"])
            .to_pandas()
            .equals(switches2.select(["eq_fk", "open"]).to_pandas())
        )
