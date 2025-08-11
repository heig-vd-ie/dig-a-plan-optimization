import json
import logging
import os
import requests
from pathlib import Path
from pipelines.expansion.models.response import ExpansionResponse
from pipelines.expansion.models.request import ExpansionRequest, Scenarios, BenderCuts


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ExpansionModel:
    def __init__(self):
        """Initialize the ExpansionModel."""
        SERVER_HOST = "localhost"

        if "SERVER_PORT" in os.environ:
            SERVER_PORT = int(os.environ["SERVER_PORT"])
        else:
            SERVER_PORT = 8080

        SERVER_BASE_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
        self.server_host = SERVER_HOST
        self.server_port = SERVER_PORT
        self.base_url = SERVER_BASE_URL
        logger.info(f"Expansion API server at: {SERVER_BASE_URL}")

    def get_server_config(self):
        """Get the server configuration."""
        return {
            "host": self.server_host,
            "port": self.server_port,
            "base_url": self.base_url,
        }

    def run_sddp_native(self, data_path: Path | None = None) -> requests.Response:
        """Run the SDDP algorithm (native implementation)."""
        try:
            data_path = (
                data_path
                if data_path
                else Path(__file__).parent.parent.parent.parent
                / "data"
                / "default.json"
            )
            logger.info(f"Using data path: {data_path}")
            with open(data_path, "r") as f:
                request_data = json.load(f)

            response = requests.patch(
                f"{self.base_url}/stochastic_planning",
                headers={"Content-Type": "application/json"},
                json=request_data,
            )
            if response.status_code != 200:
                logger.error(f"✗ Request failed with status {response.status_code}")
                logger.error(f"Response: {response.text}")
            else:
                logger.info(f"🎉 Response status: {response.status_code}")
            return response
        except FileNotFoundError:
            raise FileNotFoundError(f"✗ Could not find data file at {data_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"✗ Error parsing JSON: {e}")
        except requests.RequestException as e:
            raise ConnectionError(f"✗ Request error: {e}")
        except Exception as e:
            raise RuntimeError(f"✗ Unexpected error: {e}")

    def handle_expansion_request(
        self,
        expansion_request: ExpansionRequest,
        scenarios: Scenarios,
        bender_cuts: BenderCuts,
        cache_path: Path | None = None,
    ) -> Path:
        """Handle the expansion request."""
        if cache_path is None:
            cache_path = (
                Path(__file__).parent.parent.parent
                / ".cache"
                / "expansion_request.json"
            )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        scenarios_path = expansion_request.scenarios
        bender_cuts_path = expansion_request.planning_params.bender_cuts
        json.dump(
            expansion_request.model_dump(by_alias=True),
            open(cache_path, "w"),
            indent=4,
            ensure_ascii=False,
        )
        json.dump(
            bender_cuts.model_dump(by_alias=True),
            open(bender_cuts_path, "w"),
            indent=4,
            ensure_ascii=False,
        )
        json.dump(
            scenarios.model_dump(by_alias=True),
            open(scenarios_path, "w"),
            indent=4,
            ensure_ascii=False,
        )
        return cache_path

    def run_sddp(
        self,
        expansion_request: ExpansionRequest | None = None,
        scenarios: Scenarios | None = None,
        bender_cuts: BenderCuts | None = None,
        cache_path: Path | None = None,
    ) -> ExpansionResponse:
        """Run the SDDP algorithm."""
        if (
            (expansion_request is not None)
            and (scenarios is not None)
            and (bender_cuts is not None)
        ):
            data_path = self.handle_expansion_request(
                expansion_request, scenarios, bender_cuts, cache_path
            )
        else:
            data_path = None
        response = self.run_sddp_native(data_path)
        return ExpansionResponse(**response.json())
