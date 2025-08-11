import numpy as np
import math
import json
from pathlib import Path
from pipelines.expansion.api import run_sddp, run_sddp_native
from pipelines.expansion.models.request import (
    ExpansionRequest,
    OptimizationConfig,
    Scenarios,
    BenderCuts,
)


def test_expansion_model_native():
    response = run_sddp_native()
    assert response is not None
    assert response.status_code == 200
    assert math.isclose(
        np.mean(response.json()["objectives"]), 4848.556437980968, abs_tol=1e-5
    )
    assert math.isclose(
        np.std(response.json()["objectives"]), 4696.555097601305, abs_tol=1e-5
    )
    assert response.json()["simulations"] is not None
    assert len(response.json()["simulations"]) == 100


def test_expansion_model():
    results = run_sddp()
    assert results is not None
    assert math.isclose(np.mean(results.objectives), 4848.556437980968, abs_tol=1e-5)
    assert math.isclose(np.std(results.objectives), 4696.555097601305, abs_tol=1e-5)
    assert results.simulations is not None
    assert len(results.simulations) == 100


def test_expansion_model_with_request():
    expansion_request_data = json.load(open("data/default.json"))
    scenarios_data = json.load(open("data/scenarios.json"))
    bender_cuts_data = json.load(open("data/bender_cuts.json"))
    expansion_request_data["planning_params"][
        "bender_cuts"
    ] = ".cache/test/bender_cuts.json"
    expansion_request_data["scenarios"] = ".cache/test/scenarios.json"
    expansion_request = ExpansionRequest(
        optimization=OptimizationConfig(**expansion_request_data),
        scenarios=Scenarios(**scenarios_data),
        bender_cuts=BenderCuts(**bender_cuts_data),
    )
    results = run_sddp(
        expansion_request=expansion_request,
        cache_path=Path(".cache/test"),
    )
    assert results is not None
    assert math.isclose(np.mean(results.objectives), 4848.556437980968, abs_tol=1e-5)
    assert math.isclose(np.std(results.objectives), 4696.555097601305, abs_tol=1e-5)
    assert results.simulations is not None
    assert len(results.simulations) == 100
