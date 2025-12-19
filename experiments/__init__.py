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
from data_exporter.pp_to_dap import (
    pandapower_to_dig_a_plan_schema_with_scenarios,
)
from data_model import NodeEdgeModel

from data_display.grid_plotting import plot_grid_from_pandapower, plot_power_flow_results
from data_display.output_processing import compare_dig_a_plan_with_pandapower
from data_display.distribution_variable import DistributionVariable
from pipelines.reconfiguration import (
    DigAPlan,
    DigAPlanADMM,
    DigAPlanCombined,
    DigAPlanBender,
)
from pipelines.reconfiguration.configs import (
    BenderConfig,
    CombinedConfig,
    ADMMConfig,
    PipelineType,
)

from pipelines.reconfiguration.model_managers.admm import PipelineModelManagerADMM
from pipelines.reconfiguration.model_managers.bender import PipelineModelManagerBender
from pipelines.reconfiguration.model_managers.combined import (
    PipelineModelManagerCombined,
)
from pipelines.helpers.pyomo_utility import extract_optimization_results
from pipelines.expansion.algorithm import ExpansionAlgorithm
from plotly.subplots import make_subplots
from helper_functions import pl_to_dict, build_non_existing_dirs

from pandapower.networks import create_cigre_network_mv

from shapely import from_geojson
import joblib
from data_exporter.mock_dap import save_dap_state, load_dap_state

os.chdir(os.getcwd().replace("/src", ""))
os.environ["GRB_LICENSE_FILE"] = os.environ["HOME"] + "/gurobi_license/gurobi.lic"
