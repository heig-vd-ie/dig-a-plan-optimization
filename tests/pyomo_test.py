import pyomo.environ as pyo


if __name__=="__main__":
    
    model = pyo.AbstractModel()

    model.I = pyo.Set()

    model.L= pyo.Set()
    model.S = pyo.Set(within=model.L)
    model.F= pyo.Set(model.L, within=model.I *model.I)
    model.LF = pyo.Set(
        dimen=3, 
        initialize=lambda model: [(l, f) for l in model.L for f in model.F[l]]
    )
    
    model.nS = pyo.Set(
        dimen=1, 
        initialize=lambda model: [l for l in model.L if l not in model.S]
    )
    
    model.X = pyo.Param(model.L, default=0.0)
    model.Z = pyo.Param(model.I, default=20.0)
    model.P = pyo.Var(model.LF, domain=pyo.NonNegativeReals)
    

    # model.F = pyo.Var(model.I, domain=pyo.NonNegativeReals)

    data = {
        None: {
            'I': {None: list(range(4))},
            "L": {None: list(range(3))},
            "F": {0: [(1, 0), (0, 1)], 1: [(2, 1), (1, 2)], 2: [(3, 2), (2, 3)]},
            "S": {None: list(range(1))},
            "X": {
                0: 10,
                1: 35,
                2: 25,
            },
            'Z': {0 : 100.0, 1: 50.0, 2: 200.0}
        }
    }
    
    def test_propagation(model, l):
        left = model.F[l].first()
        right = model.F[l].last()
        return model.P[l, left] + model.P[l, right] == model.X[l]
    
    def test_propagation_2(model, l):
        left = model.F[l].first()
  
        return model.P[l, left]  ==  10
    
    def test_objective(model):
        
        return 10

    model.objective = pyo.Objective(rule=test_objective, sense=pyo.maximize)
    model.test_propagation = pyo.Constraint(model.L, rule=test_propagation)
    model.test_propagation_2 = pyo.Constraint(model.L, rule=test_propagation_2)


    instance: pyo.Model = model.create_instance(data)

    solver = pyo.SolverFactory('gurobi')

    solver.solve(instance)
    # for c in instance.component_objects(pyo.Param, active=True):
    #     print(f"Constraint: {c.name}")
    #     print(c.display())
    
    print(instance.L.display())
    print(instance.S.display())
    print(instance.nS.display())
    print(instance.P.extract_values()) # type: ignore


