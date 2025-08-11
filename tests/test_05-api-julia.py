import numpy as np
import math
import json
from pathlib import Path
from pipelines.expansion.api import ExpansionModel
from pipelines.expansion.models.request import ExpansionRequest, Scenarios, BenderCuts


def test_expansion_model_native():
    expansion_model = ExpansionModel()
    response = expansion_model.run_sddp_native()
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
    expansion_model = ExpansionModel()
    results = expansion_model.run_sddp()
    assert results is not None
    assert math.isclose(np.mean(results.objectives), 4848.556437980968, abs_tol=1e-5)
    assert math.isclose(np.std(results.objectives), 4696.555097601305, abs_tol=1e-5)
    assert results.simulations is not None
    assert len(results.simulations) == 100


def test_expansion_model_with_request():
    expansion_request_data = json.load(open("data/default.json"))
    scenarios = json.load(open("data/scenarios.json"))
    bender_cuts_data = json.load(open("data/bender_cuts.json"))
    expansion_request_data["planning_params"][
        "bender_cuts"
    ] = ".cache/test/bender_cuts.json"
    expansion_request_data["scenarios"] = ".cache/test/scenarios.json"
    expansion_model = ExpansionModel()
    results = expansion_model.run_sddp(
        expansion_request=ExpansionRequest(**expansion_request_data),
        scenarios=Scenarios(**scenarios),
        bender_cuts=BenderCuts(**bender_cuts_data),
        cache_path=Path(".cache/test/expansion_request.json"),
    )
    assert results is not None
    assert math.isclose(np.mean(results.objectives), 4848.556437980968, abs_tol=1e-5)
    assert math.isclose(np.std(results.objectives), 4696.555097601305, abs_tol=1e-5)
    assert results.simulations is not None
    assert len(results.simulations) == 100
