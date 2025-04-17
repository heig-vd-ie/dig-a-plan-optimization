import pyomo.environ as pyo

# Helper function to build common components.
def build_common_components(model):
    # === Common Sets ===
    model.I = pyo.Set()    # Set of buses
    model.L = pyo.Set()    # Set of physical lines

    # Connectivity: for each line l in L, F[l] must be provided in instance data as a set or list of candidate bus pairs.
    model.F = pyo.Set(model.L, within=model.I * model.I)

    # Candidate connectivity set LF: for each line l, include every candidate orientation (i,j)
    model.LF = pyo.Set(
        dimen=3,
        initialize=lambda m: [(l, i, j) for l in m.L for (i, j) in m.F[l]]
    )

    # Set of switchable lines (subset of L)
    model.S = pyo.Set(within=model.L)
    model.nS = pyo.Set(initialize=lambda m: [l for l in m.L if l not in m.S])

    # === Common Parameters ===
    model.r = pyo.Param(model.L)         # Resistance for each line
    model.x = pyo.Param(model.L)         # Reactance for each line
    model.p_load = pyo.Param(model.I)      # Real load at each bus
    model.q_load = pyo.Param(model.I)      # Reactive load at each bus
    model.M = pyo.Param(initialize=1e4)    # Big-M constant (adjust as needed)
