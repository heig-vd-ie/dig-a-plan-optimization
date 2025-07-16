from data_schema import NodeEdgeModel
from pipelines.data_manager import PipelineDataManager
from pipelines.result_manager import PipelineResultManager
from pipelines.configs import CombinedConfig, PipelineConfig, BenderConfig, PipelineType
from pipelines.model_managers.bender import PipelineModelManagerBender
from pipelines.model_managers.combined import PipelineModelManagerCombined


class DigAPlan:
    def __init__(self, config: PipelineConfig | BenderConfig) -> None:

        self.config = config or PipelineConfig()
        self.data_manager = PipelineDataManager(self.config.big_m, self.config.small_m)
        if (config.pipeline_type == PipelineType.BENDER) and isinstance(
            config, BenderConfig
        ):
            self.model_manager = PipelineModelManagerBender(
                config=config,
                data_manager=self.data_manager,
            )
        elif (config.pipeline_type == PipelineType.COMBINED) and isinstance(
            config, CombinedConfig
        ):
            self.model_manager = PipelineModelManagerCombined(
                config=config,
                data_manager=self.data_manager,
            )
        else:
            raise ValueError(
                f"Pipeline type {config.pipeline_type} is not supported. "
                "Please use PipelineType.BENDER or PipelineType.COMBINED."
            )
        self.result_manager = PipelineResultManager(
            data_manager=self.data_manager,
            model_manager=self.model_manager,
        )

    def add_grid_data(self, grid_data: NodeEdgeModel):
        """
        Add grid data to the model manager.
        """
        self.data_manager.add_grid_data(grid_data)
        self.model_manager.instantaniate_model(
            self.data_manager.grid_data_parameters_dict
        )

    def solve_model(self, max_iters: int = 100) -> None:
        """
        Solve the optimization model.
        """

        self.model_manager.solve_model(max_iters=max_iters)
