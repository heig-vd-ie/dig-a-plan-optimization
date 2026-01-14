from experiments import *

if __name__ == "__main__":
    payload_file = PROJECT_ROOT / "experiments/estavayer/00-reconfiguration.json"
    payload = json.load(open(payload_file, "r"))
    request = ADMMInput.model_validate(payload)

    if False:
        ## TO DEBUG LOCALLY WITHOUT API CALL ##
        from api.admm import run_admm

        results = run_admm(request)
        ######################################
    else:
        response = requests.patch(
            f"http://{LOCALHOST}:{PY_PORT}/reconfiguration/admm",
            json=payload,
        )
        print("Response status code:", response.status_code)
        print("Response JSON:", response.json())
