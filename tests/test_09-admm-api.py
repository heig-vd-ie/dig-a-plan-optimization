import os
import pytest
import json
import requests

from data_exporter.mock_dap import load_dap_state


class ExpansionApiTestBase:
    """Base class for expansion pipeline tests with common setup."""

    @pytest.fixture(autouse=True)
    def setup_common_data(self, test_basic_grid_quick_admm):
        """Set up common test data and configurations."""
        self.test_basic_grid_quick_admm = test_basic_grid_quick_admm
        with open(test_basic_grid_quick_admm) as f:
            self.payload = json.load(f)


class TestExpansionApi(ExpansionApiTestBase):
    def test_expansion_api(self):
        """Test the expansion API with the provided payload."""
        response = requests.patch(
            f"http://{os.getenv('LOCAL_HOST')}:{os.getenv('SERVER_PY_PORT', os.getenv('SERVER_PY_PORT'))}/reconfiguration/admm",
            json=self.payload,
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["switches"]) == 17
        assert len(data["voltages"]) == 32
