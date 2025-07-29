from data_schema import NodeEdgeModel
from pipelines.data_manager import PipelineDataManager
from pipelines.result_manager import PipelineResultManager
from pipelines.configs import (
    CombinedConfig,
    PipelineConfig,
    BenderConfig,
    ADMMConfig,
    PipelineType,
)
from pipelines.model_managers.bender import PipelineModelManagerBender
from pipelines.model_managers.combined import PipelineModelManagerCombined
from pipelines.model_managers.admm import PipelineModelManagerADMM


class DigAPlan:
    """
    Top‐level entrypoint for either Bender or Combined (ADMM) pipelines.
    When running a COMBINED pipeline, we loop for max_iters and do:
      1) radial subproblem (only x‐update penalty term)
      2) DistFlow subproblem (loss + penalty)
      3) z, u updates
      until convergence.
    """

    def __init__(self, config: PipelineConfig | BenderConfig | ADMMConfig) -> None:

        self.config = config or PipelineConfig()
        # Pull ρ from the config (only used in the COMBINED/ADMM pipeline)
        self.data_manager = PipelineDataManager(
            big_m=self.config.big_m,
            small_m=self.config.small_m,
            ρ=self.config.ρ,
            weight_infeasibility=self.config.weight_infeasibility,
            weight_penalty=self.config.weight_penalty,
            weight_admm_penalty=self.config.weight_admm_penalty,
        )
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
        elif (config.pipeline_type == PipelineType.ADMM) and isinstance(
            config, ADMMConfig
        ):
            self.model_manager = PipelineModelManagerADMM(
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
