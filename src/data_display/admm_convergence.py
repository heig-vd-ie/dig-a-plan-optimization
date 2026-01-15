from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from data_exporter.mock_dap import MockDigAPlan
from pipeline_reconfiguration import DigAPlanADMM
from konfig import settings


def plot_admm_convergence(
    dap: DigAPlanADMM | MockDigAPlan, base_path: Path = Path(settings.cache.figures)
) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(
        np.array(dap.model_manager.time_list[1:]) - dap.model_manager.time_list[0],
        dap.model_manager.r_norm_list,
        label="r_norm",
        marker="o",
    )
    plt.plot(
        np.array(dap.model_manager.time_list[1:]) - dap.model_manager.time_list[0],
        dap.model_manager.s_norm_list,
        label="s_norm",
        marker="o",
    )
    plt.xlabel("Seconds")
    plt.ylabel("Norm Value")
    plt.title("ADMM Iteration: r_norm and s_norm")
    plt.legend()
    plt.grid()
    plt.savefig(base_path / "admm_convergence.png")
    plt.show()
