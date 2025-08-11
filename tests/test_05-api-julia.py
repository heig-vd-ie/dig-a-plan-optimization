from pipelines.expansion.api import ExpansionModel
import numpy as np
import math


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
