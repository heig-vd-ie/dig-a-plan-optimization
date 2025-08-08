from pipelines.expansion.api import ExpansionModel
import numpy as np
import math


def test_expansion_model():
    expansion_model = ExpansionModel()
    response = expansion_model.run_sddp()
    assert response is not None
    assert response.status_code == 200
    assert math.isclose(
        np.mean(response.json()["objectives"]), 4298.449745512054, abs_tol=1e-5
    )
    assert math.isclose(
        np.std(response.json()["objectives"]), 3248.973223357565, abs_tol=1e-5
    )
    assert response.json()["simulations"] is not None
    assert len(response.json()["simulations"]) == 100
