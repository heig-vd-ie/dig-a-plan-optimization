from doctest import master
from pyomo import environ as pyo
from pyomo.environ import Suffix

from model_reconfiguration.sets import model_sets
from model_reconfiguration.parameters import model_parameters as common_parameters
from model_reconfiguration.variables import model_variables as common_variables
from model_reconfiguration.master_model.variables import master_model_variables
from model_reconfiguration.master_model.constraints import master_model_constraints
from model_reconfiguration.slave_model.parameters import slave_model_parameters
from model_reconfiguration.slave_model.variables import (
    slave_model_variables,
    infeasible_slave_model_variables,
)
from model_reconfiguration.slave_model.constraints import (
    optimal_slave_model_constraints,
    infeasible_slave_model_constraints,
)
from model_reconfiguration.combined_model.parameters import model_parameters
from model_reconfiguration.combined_model.variables import model_variables
from model_reconfiguration.combined_model.constraints import (
    combined_model_common_constraints,
    combined_model_lin_constraints,
    combined_model_constraints,
)


def generate_master_model(relaxed: bool = False) -> pyo.AbstractModel:
    master_model: pyo.AbstractModel = pyo.AbstractModel()  # type: ignore
    master_model = model_sets(master_model)
    master_model = common_parameters(master_model)
    master_model = common_variables(master_model)
    master_model = master_model_variables(master_model, relaxed=relaxed)
    master_model = master_model_constraints(master_model)
    return master_model


def generate_optimal_slave_model() -> pyo.AbstractModel:
    slave_model: pyo.AbstractModel = pyo.AbstractModel()  # type: ignore
    slave_model = model_sets(slave_model)
    slave_model = common_parameters(slave_model)
    slave_model = common_variables(slave_model)
    slave_model = slave_model_parameters(slave_model)
    slave_model = slave_model_variables(slave_model)
    slave_model = optimal_slave_model_constraints(slave_model)
    slave_model.dual = Suffix(direction=Suffix.IMPORT)
    return slave_model


def generate_infeasible_slave_model() -> pyo.AbstractModel:
    slave_model: pyo.AbstractModel = pyo.AbstractModel()  # type: ignore
    slave_model = model_sets(slave_model)
    slave_model = common_parameters(slave_model)
    slave_model = slave_model_parameters(slave_model)
    slave_model = common_variables(slave_model)
    slave_model = infeasible_slave_model_variables(slave_model)
    slave_model = infeasible_slave_model_constraints(slave_model)
    slave_model.dual = Suffix(direction=Suffix.IMPORT)
    return slave_model


def generate_combined_model() -> pyo.AbstractModel:
    """Builds the single combined radial + DistFlow model."""
    combined_model: pyo.AbstractModel = pyo.AbstractModel()  # type: ignore
    combined_model = model_sets(combined_model)
    combined_model = common_parameters(combined_model)
    combined_model = model_parameters(combined_model)
    combined_model = common_variables(combined_model)
    combined_model = model_variables(combined_model)
    combined_model = combined_model_common_constraints(combined_model)
    combined_model = combined_model_constraints(combined_model)
    combined_model.dual = Suffix(direction=Suffix.IMPORT)
    return combined_model


def generate_combined_lin_model() -> pyo.AbstractModel:
    """Builds the single combined radial + DistFlow model."""
    combined_model: pyo.AbstractModel = pyo.AbstractModel()  # type: ignore
    combined_model = model_sets(combined_model)
    combined_model = common_parameters(combined_model)
    combined_model = model_parameters(combined_model)
    combined_model = common_variables(combined_model)
    combined_model = model_variables(combined_model)
    combined_model = combined_model_common_constraints(combined_model)
    combined_model = combined_model_lin_constraints(combined_model)
    return combined_model
