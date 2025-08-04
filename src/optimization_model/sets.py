import pyomo.environ as pyo


def model_sets(model: pyo.AbstractModel) -> pyo.AbstractModel:
    model.slack_node = pyo.Set()
    model.Ω = pyo.Set()
    model.N = pyo.Set()  # Nodes indices.
    model.L = pyo.Set()  # Edges indices.
    model.S = pyo.Set(within=model.L)  # Switch indices
    model.C = pyo.Set(dimen=3, within=model.L * model.N * model.N)  # type: ignore

    model.Cs = pyo.Set(initialize=lambda m: [(l, i, j) for l, i, j in m.C if l in m.S])
    model.Cl = pyo.Set(
        initialize=lambda m: [(l, i, j) for l, i, j in m.C if l not in m.S]
    )

    model.Nes = pyo.Set(
        initialize=lambda m: [n for n in m.N if n not in m.slack_node]
    )  # Non-slack nodes

    model.CΩ = pyo.Set(
        initialize=lambda m: [(l, i, j, ω) for l, i, j in m.C for ω in m.Ω]
    )
    model.SΩ = pyo.Set(initialize=lambda m: [(l, ω) for l in m.S for ω in m.Ω])
    model.LΩ = pyo.Set(initialize=lambda m: [(l, ω) for l in m.L for ω in m.Ω])
    model.NΩ = pyo.Set(initialize=lambda m: [(n, ω) for n in m.N for ω in m.Ω])
    model.NesΩ = pyo.Set(initialize=lambda m: [(n, ω) for n in m.Nes for ω in m.Ω])
    model.slack_nodeΩ = pyo.Set(
        initialize=lambda m: [(n, ω) for n in m.slack_node for ω in m.Ω]
    )
    model.CsΩ = pyo.Set(
        initialize=lambda m: [(l, i, j, ω) for l, i, j in m.Cs for ω in m.Ω]
    )
    model.ClΩ = pyo.Set(
        initialize=lambda m: [(l, i, j, ω) for l, i, j in m.Cl for ω in m.Ω]
    )
    return model
