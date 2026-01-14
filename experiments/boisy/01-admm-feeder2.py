from experiments import *

if __name__ == "__main__":
    payload_file = PROJECT_ROOT / "experiments/boisy/00a-reconfiguration.json"
    payload = json.load(open(payload_file, "r"))
    payload["grid"]["name"] = "boisy-feeder-2"
    payload["grid"]["pp_file"] = ".cache/input/boisy/feeders/feeder_feeder_2.p"
    payload["profiles"]["load_profiles"] = [".cache/input/boisy/load_profiles/feeder_2"]
    request = ADMMInput.model_validate(payload)

    ## TO DEBUG LOCALLY WITHOUT API CALL ##
    # from api.admm import run_admm

    # results = run_admm(request)
    ######################################
    response = requests.patch(
        f"http://{LOCALHOST}:{PY_PORT}/reconfiguration/admm",
        json=payload,
    )
    print("Response status code:", response.status_code)
    print("Response JSON:", response.json())
