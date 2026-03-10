import pytest
from pathlib import Path
from api.sddp import SddpModel
from data_model.sddp import (
    SddpRequest,
    OptimizationConfig,
)
from helpers.json import load_obj_from_json


class TestExpansion:
    @pytest.fixture(autouse=True)
    def setup_common_data(self, test_cache_dir):
        self.test_cache_dir = test_cache_dir
        self.expansion_model = SddpModel()
        self.expansion_request = load_obj_from_json(
            Path(__file__).parent.parent / "examples" / "payloads_jl" / "default.json"
        )


class TestExpansionModel(TestExpansion):

    def test_expansion_model_native(self):
        response = self.expansion_model.run_sddp_native(self.expansion_request)
        assert response is not None
        assert response.status_code == 200

        assert response.json()["simulations"] is not None
        assert len(response.json()["simulations"]) == 100

    def test_expansion_model(self):
        expansion_request = SddpRequest(**self.expansion_request)
        results = self.expansion_model.run_sddp(expansion_request)
        assert results is not None
        assert results.simulations is not None
        assert len(results.simulations) == 100

    def test_expansion_model_with_request(self):
        expansion_request_data = load_obj_from_json(
            Path("examples/payloads_jl/default.json")
        )
        expansion_request = SddpRequest(
            optimization=OptimizationConfig(**expansion_request_data),
        )
        results = self.expansion_model.run_sddp(
            expansion_request=expansion_request,
        )
        assert results is not None
        assert results.simulations is not None
        assert len(results.simulations) == 100
