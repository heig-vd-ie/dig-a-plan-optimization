from experiments import *

if __name__ == "__main__":
    payload_file = PROJECT_ROOT / "experiments/boisy/00-expansion.json"
    payload = json.load(open(payload_file, "r"))

    # Optional: validate schema before sending
    request = ExpansionInput.model_validate(payload)

    if False:
        ## TO DEBUG LOCALLY WITHOUT API CALL ##
        from api.expansion import run_expansion

        results = run_expansion(request)
        ######################################
    else:
        response = requests.patch(
            f"http://{LOCALHOST}:{PY_PORT}/expansion?with_ray=true",
            json=payload,
        )
        print("Response status code:", response.status_code)
        print("Response JSON:", response.json())
