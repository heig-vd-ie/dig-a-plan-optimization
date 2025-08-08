import json
import logging
import os
import requests
from pathlib import Path


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ExpansionModel:
    def __init__(self, data_path: Path | None = None):
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
        self.data_path = (
            data_path
            if data_path
            else Path(__file__).parent.parent.parent.parent / "data" / "default.json"
        )
        logger.info(f"Using data path: {self.data_path}")

    def get_server_config(self):
        return {
            "host": self.server_host,
            "port": self.server_port,
            "base_url": self.base_url,
        }

    def run_sddp(self) -> requests.Response:
        try:
            with open(self.data_path, "r") as f:
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
            raise FileNotFoundError(f"✗ Could not find data file at {self.data_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"✗ Error parsing JSON: {e}")
        except requests.RequestException as e:
            raise ConnectionError(f"✗ Request error: {e}")
        except Exception as e:
            raise RuntimeError(f"✗ Unexpected error: {e}")
