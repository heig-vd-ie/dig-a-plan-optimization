from experiments import *

if __name__ == "__main__":
    payload_file = PROJECT_ROOT / "experiments/expansion_benchmark/00-settings.json"
    payload = json.load(open(payload_file, "r"))

    response = requests.patch(
        f"http://{LOCALHOST}:{PY_PORT}/expansion/benchmark",
        json=payload,
    )
    print("Response status code:", response.status_code)
    print("Response JSON:", response.json())
