from data_model import NodeEdgeModel
from pipeline_reconfiguration.data_manager import PipelineDataManager
from pipeline_reconfiguration.result_manager import PipelineResultManager
from data_model.reconfiguration_konfig import (
    CombinedConfig,
    PipelineConfig,
    BenderConfig,
    ADMMConfig,
)
from pipeline_reconfiguration.model_managers.bender import PipelineModelManagerBender
from pipeline_reconfiguration.model_managers.combined import (
    PipelineModelManagerCombined,
)
from pipeline_reconfiguration.model_managers.admm import PipelineModelManagerADMM


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
            ε=self.config.ε,
            ρ=self.config.ρ,
            voll=self.config.voll,
            volp=self.config.volp,
            γ_infeasibility=self.config.γ_infeasibility,
            γ_admm_penalty=self.config.γ_admm_penalty,
            γ_trafo_loss=self.config.γ_trafo_loss,
            all_scenarios=self.config.all_scenarios,
        )
        self.model_manager: (
            PipelineModelManagerBender
            | PipelineModelManagerCombined
            | PipelineModelManagerADMM
        )
        self.result_manager: PipelineResultManager

    def add_grid_data(self, grid_data: NodeEdgeModel):
        """
        Add grid data to the model manager.
        """
        self.data_manager.add_grid_data(grid_data)
        self.model_manager.instantaniate_model(
            self.data_manager.grid_data_parameters_dict
        )

    def solve_model(self, max_iters: int = 100, **kwargs) -> None:
        """
        Solve the optimization model.
        """

        self.model_manager.solve_model(max_iters=max_iters, **kwargs)


class DigAPlanBender(DigAPlan):
    """
    Entrypoint for the Bender pipeline.
    """

    def __init__(self, config: BenderConfig) -> None:
        super().__init__(config)
        self.model_manager: PipelineModelManagerBender = PipelineModelManagerBender(
            config, self.data_manager
        )
        self.result_manager = PipelineResultManager(
            data_manager=self.data_manager,
            model_manager=self.model_manager,
        )


class DigAPlanCombined(DigAPlan):
    """
    Entrypoint for the Combined pipeline.
    """

    def __init__(self, config: CombinedConfig) -> None:
        super().__init__(config)
        self.model_manager: PipelineModelManagerCombined = PipelineModelManagerCombined(
            config, self.data_manager
        )
        self.result_manager = PipelineResultManager(
            data_manager=self.data_manager,
            model_manager=self.model_manager,
        )


class DigAPlanADMM(DigAPlan):
    """
    Entrypoint for the ADMM pipeline.
    """

    def __init__(self, config: ADMMConfig) -> None:
        super().__init__(config)
        self.model_manager: PipelineModelManagerADMM = PipelineModelManagerADMM(
            config, self.data_manager
        )
        self.result_manager = PipelineResultManager(
            data_manager=self.data_manager,
            model_manager=self.model_manager,
        )
