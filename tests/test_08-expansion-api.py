import os
import pytest
import json
import requests


class ExpansionApiTestBase:
    """Base class for expansion pipeline tests with common setup."""

    @pytest.fixture(autouse=True)
    def setup_common_data(self, test_basic_grid_quick_expansion):
        """Set up common test data and configurations."""
        self.test_basic_grid_quick_expansion = test_basic_grid_quick_expansion
        with open(test_basic_grid_quick_expansion) as f:
            self.payload = json.load(f)


class TestExpansionApi(ExpansionApiTestBase):
    def test_expansion_api(self):
        """Test the expansion API with the provided payload."""
        response = requests.patch(
            f"http://localhost:{os.getenv('SERVER_PY_PORT', 8000)}/expansion",
            params={"with_ray": "true"},
            json=self.payload,
        )
        assert response.status_code == 200
        data = response.json()
        assert (
            len(data["sddp_response"]["objectives"])
            == self.payload["sddp_config"]["n_simulations"]
        )
