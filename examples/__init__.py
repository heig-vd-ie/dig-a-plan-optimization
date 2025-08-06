import os

os.chdir(os.getcwd() + "/src")

from logging import config
import pandapower as pp
import plotly.graph_objs as go
import polars as pl
from polars import col as c

from data_exporter.pandapower_to_dig_a_plan import pandapower_to_dig_a_plan_schema
from data_display.grid_plotting import plot_grid_from_pandapower
from data_display.output_processing import compare_dig_a_plan_with_pandapower
from pipelines import DigAPlan
from pipelines.configs import BenderConfig, CombinedConfig, ADMMConfig, PipelineType

from pipelines.model_managers.admm import PipelineModelManagerADMM
from pipelines.model_managers.bender import PipelineModelManagerBender
from pipelines.model_managers.combined import PipelineModelManagerCombined
from pyomo_utility import extract_optimization_results
from plotly.subplots import make_subplots
from general_function import pl_to_dict

from pandapower.networks import create_cigre_network_mv

from shapely import from_geojson
from general_function import pl_to_dict, build_non_existing_dirs

os.chdir(os.getcwd().replace("/src", ""))
os.environ["GRB_LICENSE_FILE"] = os.environ["HOME"] + "/gurobi_license/gurobi.lic"
