import json
import logging
import os
import sys
import requests
from pathlib import Path


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ExpansionModel:
    def __init__(self, data_path: Path | None = None):
        SERVER_HOST = "localhost"

        if len(sys.argv) >= 2:
            SERVER_PORT = int(sys.argv[1])
        elif "SERVER_PORT" in os.environ:
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

    def run_sddp(self) -> requests.Response | None:
        try:
            with open(self.data_path, "r") as f:
                request_data = json.load(f)

            response = requests.post(
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
            logger.error(f"✗ Could not find data file at {self.data_path}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"✗ Error parsing JSON: {e}")
            return None
        except requests.RequestException as e:
            logger.error(f"✗ Request error: {e}")
            return None
        except Exception as e:
            logger.error(f"✗ Unexpected error: {e}")
            return None


if __name__ == "__main__":
    expansion_model = ExpansionModel()
    expansion_model.run_sddp()
