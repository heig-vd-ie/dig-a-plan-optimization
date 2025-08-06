import pyomo.environ as pyo


def model_sets(model: pyo.AbstractModel) -> pyo.AbstractModel:
    model.slack_node = pyo.Set()
    model.Ω = pyo.Set()
    model.N = pyo.Set()  # Nodes indices.
    model.E = pyo.Set()  # Edges indices.
    model.S = pyo.Set(within=model.E)  # Switch indices
    model.Tr = pyo.Set(within=model.E)  # Transformer indices
    model.Taps = pyo.Set()  # Transformer tap positions
    model.C = pyo.Set(dimen=3, within=model.E * model.N * model.N)  # type: ignore

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
    model.EΩ = pyo.Set(initialize=lambda m: [(l, ω) for l in m.E for ω in m.Ω])
    model.NΩ = pyo.Set(initialize=lambda m: [(n, ω) for n in m.N for ω in m.Ω])
    model.NesΩ = pyo.Set(initialize=lambda m: [(n, ω) for n in m.Nes for ω in m.Ω])
    model.snΩ = pyo.Set(
        initialize=lambda m: [(n, ω) for n in m.slack_node for ω in m.Ω]
    )
    model.CsΩ = pyo.Set(
        initialize=lambda m: [(l, i, j, ω) for l, i, j in m.Cs for ω in m.Ω]
    )
    model.ClΩ = pyo.Set(
        initialize=lambda m: [(l, i, j, ω) for l, i, j in m.Cl for ω in m.Ω]
    )
    model.TrTapsΩ = pyo.Set(
        initialize=lambda m: [
            (tr, tap, ω) for tr in m.Tr for tap in m.Taps for ω in m.Ω
        ]
    )
    return model
