from experiments import *
import click


@click.command()
@click.option(
    "--kace",
    type=str,
    default="ieee_33",
    help="which case study, either ieee_33, boisy, estavayer or a specifc payload file",
)
@click.option(
    "--withapi",
    type=bool,
    default=True,
    help="Run with api call or debug locally without api call and ray",
)
@click.option(
    "--fixedswitches",
    type=bool,
    default=False,
    help="ADMM optimization consider switches are fixed or try to optimize it",
)
@click.option("--admmiter", type=int, help="Number of admm iteration", default=-1)
@click.option(
    "--riskmeasuretype",
    type=str,
    default="-n",
    help="""
    Risk measure type, one of the following: [Expectation, Entropic, Wasserstein, CVaR, WorstCase]
    """,
)
@click.option("--riskmeasureparam", type=int, help="Risk measure parameter", default=-1)
@click.option(
    "--cachename", type=str, help="Name of folder in cache directory", default="-n"
)
def expansion_planning_script(
    kace: str,
    withapi: bool,
    fixedswitches: bool,
    admmiter: int,
    riskmeasuretype: str,
    riskmeasureparam: int,
    cachename: str,
):
    # KACE
    match kace:
        case "boisy":
            payload_file = PROJECT_ROOT / "experiments/boisy/00-expansion.json"
        case "estavayer":
            payload_file = PROJECT_ROOT / "experiments/estavayer/00-expansion.json"
        case "ieee_33":
            payload_file = PROJECT_ROOT / "experiments/ieee_33/00-expansion.json"
        case _:
            print(
                f"none of boisy, estavayer, ieee_33 is selected, probably refers to payload_file={kace}"
            )
            payload_file = kace
    payload = json.load(open(payload_file, "r"))

    # FIXEDSWITCHES
    if fixedswitches:
        payload["admm_config"]["fixedswitches"] = True
    else:
        payload["admm_config"]["fixedswitches"] = False

    # ADMMITER
    if admmiter != -1:
        payload["admm_config"]["max_iters"] = admmiter

    # RISKMEASURETYPE
    if riskmeasuretype != "-n":
        payload["sddp_config"]["risk_measure_type"] = riskmeasuretype

    # RISKMEASUREPARAM
    if riskmeasureparam != -1:
        payload["sddp_config"]["risk_measure_param"] = riskmeasureparam

    # WITHAPI
    if not withapi:
        ## TO DEBUG LOCALLY WITHOUT API CALL ##
        from api.expansion import run_expansion

        request = ExpansionInput.model_validate(payload)
        results = run_expansion(
            request, with_ray=False, time_now=None if cachename == "-n" else cachename
        )
        print(results)
        ######################################
    else:
        response = requests.patch(
            (
                f"http://{LOCALHOST}:{PY_PORT}/expansion?with_ray=true"
                + ("" if cachename == "-n" else f"&time_now={cachename}")
            ),
            json=payload,
        )
        print("Response status code:", response.status_code)
        print("Response JSON:", response.json())


if __name__ == "__main__":
    expansion_planning_script()
