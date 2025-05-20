import pyomo.environ as pyo
from pyomo.environ import ConstraintList

def D_W_model_constraints(model: pyo.AbstractModel) -> pyo.AbstractModel:
    # objective in λ only
    model.objective = pyo.Objective(rule=D_W_obj, sense=pyo.minimize)

    # CONVEXITY: one row, rebuilt whenever model.K changes
    #model.convexity = pyo.Constraint(rule=convexity_rule)
    model.convexity = ConstraintList()

    # COUPLING: one row per (l,i,j) ∈ LC, rebuilt when K (or LC) changes
    #model.coupling  = pyo.Constraint(model.LC, rule=coupling_rule)
    model.coupling  = ConstraintList()

    return model

def D_W_obj(m):
    return sum(m.column_cost[k] * m.lambda_k[k] for k in m.K)

def convexity_rule(m):
    # skip until we have at least one column
    if not m.K:
        return pyo.Constraint.Skip
    return sum(m.lambda_k[k] for k in m.K) == 1

def coupling_rule(m, l, i, j):
    # skip until we have at least one column
    if not m.K:
        return pyo.Constraint.Skip
    return sum(m.column_d[k, (l, i, j)] * m.lambda_k[k] for k in m.K) == m.delta[l]
