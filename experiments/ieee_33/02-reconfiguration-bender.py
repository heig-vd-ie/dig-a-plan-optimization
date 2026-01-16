from experiments import *


if __name__ == "__main__":
    payload_file = PROJECT_ROOT / "experiments/ieee_33/00-reconfiguration.json"
    payload = json.load(open(payload_file, "r"))

    # Filter and adjust konfig parameters
    allowed = set(BenderConfig.model_fields.keys())
    konfig_in = payload.get("konfig", {})
    konfig_filtered = {k: v for k, v in konfig_in.items() if k in allowed}

    # Update or add any specific parameters needed for BenderConfig
    konfig_filtered.update(
        dict(
            verbose=False,
            big_m=1e2,
            factor_p=1e-3,
            factor_q=1e-3,
            factor_v=1.0,
            factor_i=1e-3,
            master_relaxed=False,

        )
    )
    payload["konfig"] = konfig_filtered

    request = BenderInput.model_validate(payload)

    if False:
        ## TO DEBUG LOCALLY WITHOUT API CALL ##
        from api.bender import run_bender

        results = run_bender(request)
        print(results)
        ######################################
    else:
        # Adjust endpoint if your API uses a different route
        response = requests.patch(
            f"http://{LOCALHOST}:{PY_PORT}/reconfiguration/bender",
            json=payload,
        )
        print("Response status code:", response.status_code)
        print("Response JSON:", response.json())
