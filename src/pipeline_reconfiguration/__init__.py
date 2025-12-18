from data_model import NodeEdgeModel4Reconfiguration
from pipelines.reconfiguration.data_manager import PipelineDataManager
from pipelines.reconfiguration.result_manager import PipelineResultManager
from data_model.reconfiguration_configs import (
    CombinedConfig,
    PipelineConfig,
    BenderConfig,
    ADMMConfig,
)
from pipelines.reconfiguration.model_managers.bender import PipelineModelManagerBender
from pipelines.reconfiguration.model_managers.combined import (
    PipelineModelManagerCombined,
)
from pipelines.reconfiguration.model_managers.admm import PipelineModelManagerADMM


class DigAPlan:
    """
    Top‐level entrypoint for either Bender or Combined (ADMM) pipelines.
    When running a COMBINED pipeline, we loop for max_iters and do:
      1) radial subproblem (only x‐update penalty term)
      2) DistFlow subproblem (loss + penalty)
      3) z, u updates
      until convergence.
    """

    def __init__(self, konfig: PipelineConfig | BenderConfig | ADMMConfig) -> None:

        self.konfig = konfig or PipelineConfig()
        # Pull ρ from the config (only used in the COMBINED/ADMM pipeline)
        self.data_manager = PipelineDataManager(
            big_m=self.konfig.big_m,
            ε=self.konfig.ε,
            ρ=self.konfig.ρ,
            voll=self.konfig.voll,
            volp=self.konfig.volp,
            γ_infeasibility=self.konfig.γ_infeasibility,
            γ_admm_penalty=self.konfig.γ_admm_penalty,
            γ_trafo_loss=self.konfig.γ_trafo_loss,
            all_scenarios=self.konfig.all_scenarios,
        )
        self.model_manager: (
            PipelineModelManagerBender
            | PipelineModelManagerCombined
            | PipelineModelManagerADMM
        )
        self.result_manager: PipelineResultManager

    def add_grid_data(self, grid_data: NodeEdgeModel4Reconfiguration):
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

    def __init__(self, konfig: BenderConfig) -> None:
        super().__init__(konfig)
        self.model_manager: PipelineModelManagerBender = PipelineModelManagerBender(
            konfig, self.data_manager
        )
        self.result_manager = PipelineResultManager(
            data_manager=self.data_manager,
            model_manager=self.model_manager,
        )


class DigAPlanCombined(DigAPlan):
    """
    Entrypoint for the Combined pipeline.
    """

    def __init__(self, konfig: CombinedConfig) -> None:
        super().__init__(konfig)
        self.model_manager: PipelineModelManagerCombined = PipelineModelManagerCombined(
            konfig, self.data_manager
        )
        self.result_manager = PipelineResultManager(
            data_manager=self.data_manager,
            model_manager=self.model_manager,
        )


class DigAPlanADMM(DigAPlan):
    """
    Entrypoint for the ADMM pipeline.
    """

    def __init__(self, konfig: ADMMConfig) -> None:
        super().__init__(konfig)
        self.model_manager: PipelineModelManagerADMM = PipelineModelManagerADMM(
            konfig, self.data_manager
        )
        self.result_manager = PipelineResultManager(
            data_manager=self.data_manager,
            model_manager=self.model_manager,
        )
