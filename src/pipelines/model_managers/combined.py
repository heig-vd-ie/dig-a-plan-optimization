import random
import pyomo.environ as pyo
import polars as pl
from general_function import generate_log
from pipelines.data_manager import PipelineDataManager
from pipelines.configs import CombinedConfig, PipelineType
from optimization_model import generate_combined_model, generate_combined_lin_model
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
        self.combined_lin_model: pyo.AbstractModel = generate_combined_lin_model()
        self.combined_model_instance: pyo.ConcreteModel
        self.combined_lin_model_instance: pyo.ConcreteModel

        self.combined_obj: float = 0.0

        self.combined_obj_list: list[float] = []

    def instantaniate_model(self, grid_data_parameters_dict: dict | None) -> None:
        self.combined_model_instance = self.combined_model.create_instance(grid_data_parameters_dict[list(grid_data_parameters_dict.keys())[0]])  # type: ignore
        self.combined_lin_model_instance = self.combined_lin_model.create_instance(grid_data_parameters_dict[list(grid_data_parameters_dict.keys())[0]])  # type: ignore

    def solve_model(self, groups: int | None = None, **kwargs) -> None:
        """Solve the combined radial+DistFlow model."""
        results = self.solver.solve(
            self.combined_lin_model_instance, tee=self.config.verbose
        )

        δ_map = self.combined_lin_model_instance.δ.extract_values()  # type: ignore

        if groups is not None:
            switch_list = list(self.combined_model_instance.S)  # type: ignore

            random_switches = random.sample(
                switch_list,
                k=min(
                    max(2, int(len(switch_list) / groups)),
                    len(switch_list) - 1,
                ),
            )
            for edge_id, δ in δ_map.items():
                if edge_id in random_switches:
                    continue
                self.combined_model_instance.δ[edge_id].fix(δ)  # type: ignore

        results = self.solver.solve(
            self.combined_model_instance, tee=self.config.verbose
        )

        if results.solver.termination_condition != pyo.TerminationCondition.optimal:
            log.error(f"Solve failed: {results.solver.termination_condition}")
            return
        current_obj = pyo.value(self.combined_model_instance.objective)
        self.combined_obj_list.append(current_obj)  # type: ignore
        log.info(f"Combined solve successful: objective = {current_obj:.4f}")
