# %%
import pyomo.environ as pyo
import pyomo.kernel as pmo
import polars as pl
from typing import Union
from polars import col as c

# %%
m = pmo.block()

m.idx = pmo.set(initialize=range(3))  # type: ignore
m.P = pmo.variable_dict()
m.Q = pmo.variable_dict()
m.I = pmo.variable_dict()
m.U = pmo.parameter_dict()


for i in m.idx:  # type: ignore
    m.P[i] = pmo.variable(domain=pyo.NonNegativeReals)  # type: ignore
    m.Q[i] = pmo.variable(domain=pyo.NonNegativeReals)  # type: ignore
    m.I[i] = pmo.variable(domain=pyo.NonNegativeReals)  # type: ignore
    m.U[i] = pmo.parameter(3 + 5 * i)  # type: ignore

m.power = pmo.constraint_dict()
m.power_2 = pmo.constraint_dict()


def socp_relaxation(m, P, Q, I, U, idx):
    m.ul_sum = pmo.variable_dict()
    m.ul_diff = pmo.variable_dict()
    m.conic_power = pmo.constraint_dict()
    m.ul_sum_value = pmo.constraint_dict()
    m.ul_diff_value = pmo.constraint_dict()
    for i in idx:
        m.ul_sum[i] = pmo.variable(domain=pyo.NonNegativeReals)
        m.ul_diff[i] = pmo.variable(domain=pyo.Reals)
        m.ul_sum_value[i] = pmo.constraint(m.ul_sum[i] == (I[i] + m.U[i]) / 2)
        m.ul_diff_value[i] = pmo.constraint(m.ul_diff[i] == (I[i] - U[i]) / 2)
        m.conic_power[i] = pmo.conic.quadratic(
            x=[P[i], Q[i], m.ul_diff[i]], r=m.ul_sum[i]
        )


for i in m.idx:  # type: ignore
    m.power[i] = pmo.constraint(m.P[i] == 2 * i + 2)  # type: ignore
    m.power_2[i] = pmo.constraint(m.Q[i] == 3 * i)  # type: ignore

socp_relaxation(m=m, P=m.P, Q=m.Q, I=m.I, U=m.U, idx=m.idx)  # type: ignore

m.o = pmo.objective(sum(m.I[i] ** 2 for i in m.idx), sense=pyo.minimize)  # type: ignore

solver = pmo.SolverFactory("gurobi")

results = solver.solve(m, tee=False)  # tee=True to see solver output


# %%
def get_vars_results(
    key: list, var_list: list[Union[pmo.variable_dict, pmo.parameter_dict]]
):
    results = []

    for i in key:
        results.append(
            dict(
                map(
                    lambda var: (
                        (var.name, var[i]()) if i in var.keys() else (var.name, None)
                    ),
                    var_list,
                )
            )
        )
    return pl.from_dicts(results).select(pl.Series(key).alias("idx"), pl.all())


results = get_vars_results(m.idx, [m.P, m.Q, m.U, m.I])  # type: ignore

results.with_columns(((c("P").pow(2) + c("Q").pow(2)) / c("U")).alias("I_Calc"))
