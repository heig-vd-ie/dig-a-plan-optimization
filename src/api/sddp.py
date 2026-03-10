import json
import os
import requests
from typing import Dict
from data_model.sddp import (
    SddpResponse,
    Scenarios,
    SddpRequest,
    LongTermScenarioRequest,
)
from helpers import generate_log

log = generate_log(name=__name__)

SERVER_BASE_URL = f"http://{os.environ.get("LOCAL_HOST", "localhost")}:{os.environ.get("SERVER_JL_PORT", 8082)}"


class SddpModel:
    def __init__(self):
        """Initialize the SddpModel."""
        self.base_url = SERVER_BASE_URL
        log.info(f"SDDP API server at: {SERVER_BASE_URL}")

    def run_sddp_native(self, request_data: dict) -> requests.Response:
        """Run the SDDP algorithm (native implementation)."""
        try:
            response = requests.patch(
                f"{self.base_url}/stochastic_planning",
                headers={"Content-Type": "application/json"},
                json=request_data,
            )
            if response.status_code != 200:
                log.error(f"✗ Request failed with status {response.status_code}")
                log.error(f"Response: {response.text}")
            else:
                log.info(f"🎉 Response status: {response.status_code}")
            return response
        except json.JSONDecodeError as e:
            raise ValueError(f"✗ Error parsing JSON: {e}")
        except requests.RequestException as e:
            raise ConnectionError(f"✗ Request error: {e}")
        except Exception as e:
            raise RuntimeError(f"✗ Unexpected error: {e}")

    def run_generate_scenarios_native(self, request_data: Dict) -> requests.Response:
        """Run the SDDP algorithm (native implementation)."""
        try:
            response = requests.patch(
                f"{self.base_url}/generate-scenarios",
                headers={"Content-Type": "application/json"},
                json=request_data,
            )
            if response.status_code != 200:
                log.error(f"✗ Request failed with status {response.status_code}")
                log.error(f"Response: {response.text}")
            else:
                log.info(f"🎉 Response status: {response.status_code}")
            return response
        except json.JSONDecodeError as e:
            raise ValueError(f"✗ Error parsing JSON: {e}")
        except requests.RequestException as e:
            raise ConnectionError(f"✗ Request error: {e}")
        except Exception as e:
            raise RuntimeError(f"✗ Unexpected error: {e}")

    def run_sddp(self, expansion_request: SddpRequest) -> SddpResponse:
        """Run the SDDP algorithm."""
        response = self.run_sddp_native(
            request_data=expansion_request.optimization.model_dump(
                by_alias=True, mode="json"
            )
        )
        return SddpResponse(**response.json())

    def run_generate_scenarios(
        self, long_term_scenario_request: LongTermScenarioRequest
    ) -> Scenarios:
        response = self.run_generate_scenarios_native(
            request_data=long_term_scenario_request.model_dump(by_alias=True)
        )
        return Scenarios(**response.json())
