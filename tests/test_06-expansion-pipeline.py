import pandapower as pp
from data_exporter.pandapower_to_dig_a_plan import pandapower_to_dig_a_plan_schema
from data_exporter.dig_a_plan_to_expansion import dig_a_plan_to_expansion
from pathlib import Path
from pipelines.expansion.models.request import (
    PlanningParams,
    AdditionalParams,
    BenderCuts,
    RiskMeasureType,
    Scenarios,
)
from pipelines.expansion.ltscenarios import generate_long_term_scenarios
from pipelines.expansion.api import run_sddp


def test_expansion_data_exporter():

    net = pp.from_pickle("data/simple_grid.p")
    grid_data = pandapower_to_dig_a_plan_schema(net)

    planning_params = PlanningParams(
        n_stages=3,
        initial_budget=100000,
        discount_rate=0.05,
    )
    additional_params = AdditionalParams(
        iteration_limit=10,
        n_simulations=100,
        risk_measure_type=RiskMeasureType.EXPECTATION,
        risk_measure_param=0.1,
        seed=42,
    )
    scenario_data = generate_long_term_scenarios(
        nodes=grid_data.node_data,
        δ_load_var=0.1,
        δ_pv_var=0.1,
        δ_b_var=0.1,
        number_of_scenarios=100,
        number_of_stages=3,
        seed_number=42,
    )
    expansion_request = dig_a_plan_to_expansion(
        grid_data=grid_data,
        planning_params=planning_params,
        additional_params=additional_params,
        scenarios_data=scenario_data,
        bender_cuts=BenderCuts(cuts={}),
        scenarios_cache=Path(".cache/test/scenarios.json"),
        bender_cuts_cache=Path(".cache/test/bender_cuts.json"),
    )

    results = run_sddp(expansion_request, Path(".cache/test"))

    assert results is not None


# from pipelines.reconfiguration.configs import ADMMConfig
# from pipelines.reconfiguration import DigAPlanADMM
# from pipelines.reconfiguration.model_managers import PipelineType, bender
# groups = {
#     0: [19, 20, 21, 29, 32, 35],
#     1: [35, 30, 33, 25, 26, 27],
#     2: [27, 32, 22, 23, 34],
#     3: [31, 24, 28, 21, 22, 23],
#     4: [34, 26, 25, 24, 31],
# }

# config = ADMMConfig(
#     verbose=False,
#     pipeline_type=PipelineType.ADMM,
#     solver_name="gurobi",
#     solver_non_convex=2,
#     big_m=1e3,
#     ε=1,
#     ρ=2.0,
#     γ_infeasibility=100.0,
#     γ_admm_penalty=1.0,
#     groups=groups,
#     max_iters=10,
#     μ=10.0,
#     τ_incr=2.0,
#     τ_decr=2.0,
# )

# dap = DigAPlanADMM(config=config)
# dap.add_grid_data(grid_data)

# dap.model_manager.solve_model()
