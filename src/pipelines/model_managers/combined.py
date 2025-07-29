import pyomo.environ as pyo
import polars as pl
from general_function import generate_log
from pipelines.data_manager import PipelineDataManager
from pipelines.configs import CombinedConfig, PipelineType
from optimization_model import generate_combined_model
from pipelines.model_managers import PipelineModelManager

log = generate_log(name=__name__)


class PipelineModelManagerCombined(PipelineModelManager):
    def __init__(
        self,
        config: CombinedConfig,
        data_manager: PipelineDataManager,
        pipeline_type=PipelineType.COMBINED,
    ) -> None:
        """Initialize the combined model manager with configuration and data manager."""
        super().__init__(config, data_manager, pipeline_type)

        self.combined_model: pyo.AbstractModel = generate_combined_model()
        self.combined_model_instance: pyo.ConcreteModel
        self.scaled_combined_model_instance: pyo.ConcreteModel

        self.combined_obj: float = 0.0

        self.combined_obj_list: list[float] = []

    def instantaniate_model(self, grid_data_parameters_dict: dict | None) -> None:
        self.combined_model_instance = self.combined_model.create_instance(grid_data_parameters_dict)  # type: ignore
        self.δ_variable = pl.DataFrame(
            self.combined_model_instance.δ.items(),  # type: ignore
            schema=["S", "delta_variable"],
        )

    def solve_model(self, **kwargs) -> None:
        """Solve the combined radial+DistFlow model."""
        results = self.solver.solve(
            self.combined_model_instance, tee=self.config.verbose
        )
        if results.solver.termination_condition != pyo.TerminationCondition.optimal:
            log.error(f"Solve failed: {results.solver.termination_condition}")
            return
        current_obj = pyo.value(self.combined_model_instance.objective)
        self.combined_obj_list.append(current_obj)  # type: ignore
        log.info(f"Combined solve successful: objective = {current_obj:.4f}")
        self.δ_variable = pl.DataFrame(
            self.combined_model_instance.δ.items(),  # type: ignore
            schema=["S", "delta_variable"],
        )
