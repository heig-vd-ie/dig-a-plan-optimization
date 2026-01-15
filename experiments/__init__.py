import os
import numpy as np
import json
import requests
from pathlib import Path
from logging import config
import copy
import pandapower as pp
import plotly.graph_objs as go
import polars as pl
from polars import col as c
from datetime import datetime
from konfig import settings
from api.grid_cases import (
    get_grid_case,
)
from data_model import NodeEdgeModel

from data_display.grid_plotting import (
    plot_grid_from_pandapower,
    plot_power_flow_results,
)
from data_display.output_processing import compare_dig_a_plan_with_pandapower
from data_display.distribution_variable import DistributionVariable
from pipeline_reconfiguration import (
    DigAPlan,
    DigAPlanADMM,
    DigAPlanCombined,
    DigAPlanBender,
)
from data_model.reconfiguration import (
    BenderConfig,
    CombinedConfig,
    ADMMConfig,
)
from data_model.expansion import SDDPConfig, LongTermUncertainty

from pipeline_reconfiguration.model_managers.admm import PipelineModelManagerADMM
from pipeline_reconfiguration.model_managers.bender import PipelineModelManagerBender
from pipeline_reconfiguration.model_managers.combined import (
    PipelineModelManagerCombined,
)
from helpers.pyomo import extract_optimization_results
from pipeline_expansion.algorithm import ExpansionAlgorithm
from plotly.subplots import make_subplots
from helpers import pl_to_dict, build_non_existing_dirs

from pandapower.networks import create_cigre_network_mv

from shapely import from_geojson
import joblib
from data_exporter.mock_dap import save_dap_state, load_dap_state
from data_model.reconfiguration import GridCaseModel
from data_model import ShortTermUncertaintyRandom
from api.combined import CombinedInput
from api.admm import ADMMInput
from helpers.json import load_obj_from_json
from data_display.admm_convergence import plot_admm_convergence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCALHOST = os.getenv("LOCAL_HOST")
PY_PORT = os.getenv("SERVER_PY_PORT")
OUTPUT_ADMM_PATH = Path(settings.cache.outputs_admm)
