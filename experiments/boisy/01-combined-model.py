from experiments import *

if __name__ == "__main__":
    payload_file = PROJECT_ROOT / "experiments/boisy/00a-reconfiguration.json"
    payload = json.load(open(payload_file, "r"))
    request = CombinedInput.model_validate(payload)

    ## TO DEBUG LOCALLY WITHOUT API CALL ##
    # from api.combined import run_combined

    # request.profiles = None  # to speed up local testing
    # results = run_combined(request)
    ######################################
    payload["profiles"] = None
    response = requests.patch(
        f"http://{LOCALHOST}:{PY_PORT}/reconfiguration/combined",
        json=payload,
    )
    print("Response status code:", response.status_code)
    print("Response JSON:", response.json())
