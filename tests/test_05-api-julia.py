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
        self.sddp_model = SddpModel()
        self.sddp_request = SddpRequest(
            optimization=OptimizationConfig(
                **load_obj_from_json(
                    Path(__file__).parent.parent
                    / "examples"
                    / "payloads_jl"
                    / "default.json"
                )
            )
        )
        self.sddp_request_native = self.sddp_request.model_dump(
            by_alias=True, mode="json"
        )


class TestExpansionModel(TestExpansion):

    def test_expansion_model_native(self):
        response = self.sddp_model.run_sddp_native(self.sddp_request_native)
        assert response is not None
        assert response.status_code == 200

        assert response.json()["simulations"] is not None
        assert len(response.json()["simulations"]) == 100

    def test_expansion_model(self):
        results = self.sddp_model.run_sddp(self.sddp_request)
        assert results is not None
        assert results.simulations is not None
        assert len(results.simulations) == 100

    def test_expansion_model_with_request(self):
        results = self.sddp_model.run_sddp(
            expansion_request=self.sddp_request,
        )
        assert results is not None
        assert results.simulations is not None
        assert len(results.simulations) == 100
