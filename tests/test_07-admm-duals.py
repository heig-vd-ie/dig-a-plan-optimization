import pytest
import polars as pl
from api.grid_cases import get_grid_case
from pipeline_reconfiguration import DigAPlanADMM
from data_model.reconfiguration import ADMMConfig


class ExpansionTestBase:
    """Base class for expansion pipeline tests with common setup."""

    @pytest.fixture(autouse=True)
    def setup_common_data(
        self,
        test_simple_grid,
        test_admm_config,
        test_simple_grid_groups,
        test_seed,
        test_short_term_uncertainty_random,
    ):
        """Set up common test data and configurations."""
        self.grid = test_simple_grid
        self.seed = test_seed
        self.stu = test_short_term_uncertainty_random
        self.net, self.grid_data = get_grid_case(
            grid=self.grid, seed=self.seed, stu=self.stu
        )
        self.admm_config: ADMMConfig = test_admm_config
        self.simple_grid_groups = test_simple_grid_groups


class TestADMMModelDualExample(ExpansionTestBase):
    def test_extract_duals_simple_admm(self):
        """Extract dual variables from a simple ADMM problem."""
        konfig = self.admm_config
        konfig.groups = self.simple_grid_groups
        self.grid_data.edge_data = self.grid_data.edge_data.with_columns(
            pl.when(pl.col("type") == "transformer")
            .then(list(range(95, 105, 1)))
            .otherwise([100])
            .alias("taps")
        )
        dap = DigAPlanADMM(konfig=konfig)
        dap.add_grid_data(self.grid_data)
        dap.model_manager.solve_model(extract_duals=True)
        duals = dap.result_manager.extract_dual_variables(scenario=0)
        assert duals.shape[0] == 933
        duals = dap.result_manager.extract_duals_for_expansion()
        assert duals.shape[0] == 830
        assert duals.shape[1] == 4
        θs = dap.result_manager.extract_reconfiguration_θ()
        assert θs.shape[0] == 10
        assert θs.shape[1] == 2
        assert θs["θ"].sum() == pytest.approx(0.0, abs=1e-1)
