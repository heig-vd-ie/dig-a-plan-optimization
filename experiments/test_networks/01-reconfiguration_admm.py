from experiments import *

if __name__ == "__main__":
    payload_file = PROJECT_ROOT / "experiments/test_networks/00-reconfiguration.json"
    payload = json.load(open(payload_file, "r"))

    # Optional: validate schema before sending 
    request = ADMMInput.model_validate(payload)

    response = requests.patch(
        f"http://{LOCALHOST}:{PY_PORT}/reconfiguration/admm",
        json=payload,
    )
    print("Response status code:", response.status_code)
    print("Response JSON:", response.json())