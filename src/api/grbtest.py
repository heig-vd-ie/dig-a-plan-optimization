import gurobipy as gp
from gurobipy import GRB
import sys

print("=" * 60)
print("GUROBI LICENSE VERIFICATION TEST")
print("=" * 60)

try:
    # Test 1: Check Gurobi version
    print(f"\n1. Gurobi Version: {gp.gurobi.version()}")
    
    # Test 2: Create environment and check license info
    print("\n2. Creating Gurobi Environment...")
    env = gp.Env()
    print("   ✓ Environment created successfully!")
    print("   (If you see a license message above, that's your license info)")
    
    # Test 3: Create and solve a simple model
    print("\n3. Testing Model Solving...")
    model = gp.Model(env=env)
    
    # Create a trivial model
    x = model.addVar(name="x")
    model.setObjective(x, GRB.MINIMIZE)
    model.addConstr(x >= 1, "c0")
    
    # Solve
    model.setParam('OutputFlag', 0)  # Suppress output
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        print(f"   ✓ Model solved successfully!")
        print(f"   Optimal value: {model.objVal}")
    else:
        print(f"   ✗ Model solve failed with status: {model.status}")
    
    # Cleanup
    model.dispose()
    env.dispose()
    
    print("\n" + "=" * 60)
    print("SUCCESS: Your Gurobi license is working correctly!")
    print("=" * 60)

except gp.GurobiError as e:
    print("\n" + "=" * 60)
    print(f"GUROBI ERROR: {e.message}")
    print("=" * 60)
    print("\nPossible solutions:")
    print("1. Restart your Python session/kernel completely")
    print("2. Check your gurobi.lic file exists and is readable")
    print("3. Verify internet connection to token.gurobi.com:443")
    print("4. Wait a bit longer and try again (caching issue)")
    print("5. Try deleting Python cache: __pycache__ folders")
    sys.exit(1)

except Exception as e:
    print("\n" + "=" * 60)
    print(f"UNEXPECTED ERROR: {str(e)}")
    print("=" * 60)
    sys.exit(1)
