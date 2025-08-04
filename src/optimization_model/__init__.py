from pyomo import environ as pyo
from pyomo.environ import Suffix

from optimization_model.master_model.sets import master_model_sets
from optimization_model.master_model.parameters import master_model_parameters
from optimization_model.master_model.variables import master_model_variables
from optimization_model.master_model.constraints import master_model_constraints
from optimization_model.slave_model.sets import slave_model_sets
from optimization_model.slave_model.parameters import slave_model_parameters
from optimization_model.slave_model.variables import (
    slave_model_variables,
    infeasible_slave_model_variables,
)
from optimization_model.slave_model.constraints import (
    optimal_slave_model_constraints,
    infeasible_slave_model_constraints,
)
from optimization_model.combined_model.sets import model_sets
from optimization_model.combined_model.parameters import model_parameters
from optimization_model.combined_model.variables import model_variables
from optimization_model.combined_model.constraints import (
    combined_model_common_constraints,
    combined_model_lin_constraints,
    combined_model_constraints,
)


def generate_master_model(relaxed: bool = False) -> pyo.AbstractModel:
    master_model: pyo.AbstractModel = pyo.AbstractModel()  # type: ignore
    master_model = master_model_sets(master_model)
    master_model = master_model_parameters(master_model)
    master_model = master_model_variables(master_model, relaxed=relaxed)
    master_model = master_model_constraints(master_model)
    return master_model


def generate_optimal_slave_model() -> pyo.AbstractModel:
    slave_model: pyo.AbstractModel = pyo.AbstractModel()  # type: ignore
    slave_model = slave_model_sets(slave_model)
    slave_model = slave_model_parameters(slave_model)
    slave_model = slave_model_variables(slave_model)
    slave_model = optimal_slave_model_constraints(slave_model)
    slave_model.dual = Suffix(direction=Suffix.IMPORT)
    return slave_model


def generate_infeasible_slave_model() -> pyo.AbstractModel:
    slave_model: pyo.AbstractModel = pyo.AbstractModel()  # type: ignore
    slave_model = slave_model_sets(slave_model)
    slave_model = slave_model_parameters(slave_model)
    slave_model = infeasible_slave_model_variables(slave_model)
    slave_model = infeasible_slave_model_constraints(slave_model)
    slave_model.dual = Suffix(direction=Suffix.IMPORT)
    return slave_model


def generate_combined_model() -> pyo.AbstractModel:
    """Builds the single combined radial + DistFlow model."""
    combined_model: pyo.AbstractModel = pyo.AbstractModel()  # type: ignore
    combined_model = model_sets(combined_model)
    combined_model = model_parameters(combined_model)
    combined_model = model_variables(combined_model)
    combined_model = combined_model_common_constraints(combined_model)
    combined_model = combined_model_constraints(combined_model)
    return combined_model


def generate_combined_lin_model() -> pyo.AbstractModel:
    """Builds the single combined radial + DistFlow model."""
    combined_model: pyo.AbstractModel = pyo.AbstractModel()  # type: ignore
    combined_model = model_sets(combined_model)
    combined_model = model_parameters(combined_model)
    combined_model = model_variables(combined_model)
    combined_model = combined_model_common_constraints(combined_model)
    combined_model = combined_model_lin_constraints(combined_model)
    return combined_model
