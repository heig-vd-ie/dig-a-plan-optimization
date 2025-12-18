import os

os.chdir(os.getcwd() + "/src")
import numpy as np
from pathlib import Path
from logging import config
import copy
import pandapower as pp
import plotly.graph_objs as go
import polars as pl
from polars import col as c
from datetime import datetime
from config import settings
from data_exporter.kace_to_dap import (
    kace4reconfiguration,
    kace4expansion,
)
from data_model import NodeEdgeModel4Reconfiguration
from data_display.grid_plotting import (
    plot_grid_from_pandapower,
    plot_power_flow_results,
)
from data_display.output_processing import compare_dig_a_plan_with_pandapower
from data_display.distribution_variable import DistributionVariable
from pipelines.reconfiguration import (
    DigAPlan,
    DigAPlanADMM,
    DigAPlanCombined,
    DigAPlanBender,
)
from data_model.reconfiguration_configs import (
    BenderConfig,
    CombinedConfig,
    ADMMConfig,
)
from pipelines.reconfiguration.model_managers.admm import PipelineModelManagerADMM
from pipelines.reconfiguration.model_managers.bender import PipelineModelManagerBender
from pipelines.reconfiguration.model_managers.combined import (
    PipelineModelManagerCombined,
)
from pipelines.helpers.json_rw import load_obj_from_json
from data_model.reconfiguration import BenderInput
from pipelines.helpers.pyomo_utility import extract_optimization_results
from pipelines.expansion.algorithm import ExpansionAlgorithm
from plotly.subplots import make_subplots
from helper_functions import pl_to_dict, build_non_existing_dirs
from pandapower.networks import create_cigre_network_mv
from shapely import from_geojson
import joblib
from data_exporter.dap_to_mock import save_dap_state, load_dap_state

os.chdir(os.getcwd().replace("/src", ""))
os.environ["GRB_LICENSE_FILE"] = os.environ["HOME"] + "/gurobi_license/gurobi.lic"


__all__ = [
    "os",
    "np",
    "Path",
    "config",
    "copy",
    "pp",
    "go",
    "pl",
    "c",
    "datetime",
    "settings",
    "load_obj_from_json",
    "kace4reconfiguration",
    "kace4expansion",
    "NodeEdgeModel4Reconfiguration",
    "plot_grid_from_pandapower",
    "plot_power_flow_results",
    "compare_dig_a_plan_with_pandapower",
    "BenderInput",
    "DistributionVariable",
    "DigAPlan",
    "DigAPlanADMM",
    "DigAPlanCombined",
    "DigAPlanBender",
    "BenderConfig",
    "CombinedConfig",
    "ADMMConfig",
    "PipelineModelManagerADMM",
    "PipelineModelManagerBender",
    "PipelineModelManagerCombined",
    "extract_optimization_results",
    "ExpansionAlgorithm",
    "make_subplots",
    "pl_to_dict",
    "build_non_existing_dirs",
    "create_cigre_network_mv",
    "from_geojson",
    "joblib",
    "save_dap_state",
    "load_dap_state",
]
