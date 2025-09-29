import json
import logging
import os
import requests
from pathlib import Path
from pipelines.expansion.models.response import ExpansionResponse
from pipelines.expansion.models.request import ExpansionRequest
from pipelines.helpers.json_rw import save_obj_to_json, load_obj_from_json

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
            SERVER_PORT = 8081

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

    def run_sddp_native(
        self,
        request_data: dict | None = None,
        data_path: Path | None = None,
    ) -> requests.Response:
        """Run the SDDP algorithm (native implementation)."""
        try:
            if request_data is None:
                data_path = (
                    data_path
                    if data_path
                    else Path(__file__).parent.parent.parent.parent
                    / "data"
                    / "default.json"
                )
                logger.info(f"Using data path: {data_path}")
                request_data = load_obj_from_json(data_path)

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
        cache_path: Path | None = None,
    ) -> Path:
        """Handle the expansion request."""
        if cache_path is None:
            cache_path = Path(__file__).parent.parent.parent / ".cache"
        cache_path.mkdir(parents=True, exist_ok=True)
        scenarios_path = expansion_request.optimization.scenarios
        out_of_sample_scenarios_path = (
            expansion_request.optimization.out_of_sample_scenarios
        )
        bender_cuts_path = expansion_request.optimization.bender_cuts
        save_obj_to_json(expansion_request.bender_cuts, Path(bender_cuts_path))
        save_obj_to_json(expansion_request.scenarios, Path(scenarios_path))
        save_obj_to_json(
            expansion_request.out_of_sample_scenarios,
            Path(out_of_sample_scenarios_path),
        )
        return cache_path

    def run_sddp(
        self,
        expansion_request: ExpansionRequest | None = None,
        cache_path: Path | None = None,
    ) -> ExpansionResponse:
        """Run the SDDP algorithm."""
        if expansion_request is not None:
            cache_path = self.handle_expansion_request(expansion_request, cache_path)
            response = self.run_sddp_native(
                request_data=expansion_request.optimization.model_dump(by_alias=True)
            )
        else:
            response = self.run_sddp_native()
        return ExpansionResponse(**response.json())


def run_sddp_native(data_path: Path | None = None) -> requests.Response:
    expansion_model = ExpansionModel()
    return expansion_model.run_sddp_native(data_path=data_path)


def run_sddp(
    expansion_request: ExpansionRequest | None = None, cache_path: Path | None = None
) -> ExpansionResponse:
    expansion_model = ExpansionModel()
    return expansion_model.run_sddp(
        expansion_request=expansion_request,
        cache_path=cache_path,
    )
