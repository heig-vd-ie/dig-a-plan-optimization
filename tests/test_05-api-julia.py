import pytest
import numpy as np
import math
from pathlib import Path
from api.sddp import run_sddp, run_sddp_native
from data_model.sddp import (
    ExpansionRequest,
    OptimizationConfig,
    Scenarios,
    BenderCuts,
)
from helpers.json import load_obj_from_json


class TestExpansion:
    @pytest.fixture(autouse=True)
    def setup_common_data(self, test_cache_dir):
        self.test_cache_dir = test_cache_dir


class TestExpansionModel(TestExpansion):

    def test_expansion_model_native(self):
        response = run_sddp_native()
        assert response is not None
        assert response.status_code == 200

        assert response.json()["simulations"] is not None
        assert len(response.json()["simulations"]) == 100

    def test_expansion_model(self):
        results = run_sddp()
        assert results is not None
        assert results.simulations is not None
        assert len(results.simulations) == 100

    def test_expansion_model_with_request(self):
        expansion_request_data = load_obj_from_json(
            Path("examples/payloads_jl/default.json")
        )
        scenarios_data = load_obj_from_json(Path("examples/payloads_jl/scenarios.json"))
        out_of_sample_scenarios_data = load_obj_from_json(
            Path("examples/payloads_jl/out_of_sample_scenarios.json")
        )
        bender_cuts_data = load_obj_from_json(
            Path("examples/payloads_jl/bender_cuts.json")
        )
        expansion_request = ExpansionRequest(
            optimization=OptimizationConfig(**expansion_request_data),
            scenarios=Scenarios(**scenarios_data),
            out_of_sample_scenarios=Scenarios(**out_of_sample_scenarios_data),
            bender_cuts=BenderCuts(**bender_cuts_data),
        )
        results = run_sddp(
            expansion_request=expansion_request,
            cache_path=self.test_cache_dir,
        )
        assert results is not None
        assert results.simulations is not None
        assert len(results.simulations) == 100
