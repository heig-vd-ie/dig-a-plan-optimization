from experiments import *

if __name__ == "__main__":
    payload_file = PROJECT_ROOT / "experiments/ieee_33/00-reconfiguration.json"
    payload = json.load(open(payload_file, "r"))

    # Filter and adjust konfig parameters
    allowed = set(CombinedConfig.model_fields.keys())
    konfig_in = payload.get("konfig", {})
    konfig_filtered = {k: v for k, v in konfig_in.items() if k in allowed}

    # Update parameters needed for CombinedConfig
    konfig_filtered.update(
        dict(
            verbose=True,
            threads=1,
            big_m=1e2,
            γ_infeasibility=1.0,
            factor_v=1.0,
            factor_i=1e-3,
        )
    )

    payload["konfig"] = konfig_filtered

    # Validate schema before sending
    request = CombinedInput.model_validate(payload)

    if False:
        ## TO DEBUG LOCALLY WITHOUT API CALL ##
        from api.combined import run_combined

        results = run_combined(request)
        LOG.info(results)
        ######################################
    else:
        response = requests.patch(
            f"http://{LOCALHOST}:{PY_PORT}/reconfiguration/combined",
            json=payload,
        )
        LOG.info(f"Response status code: {response.status_code}")
        try:
            LOG.info(f"Response JSON: {response.json()}")
        except Exception:
            LOG.info(f"Response text: {response.text}")
