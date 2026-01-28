from api.bender import run_bender
from api.combined import run_combined
from api.admm import run_admm
from api.expansion import run_expansion
from api.ray_utils import init_ray, shutdown_ray, where_am_i
from data_model.reconfiguration import (
    ADMMInput,
    CombinedInput,
    BenderInput,
    ReconfigurationOutput,
)
from data_model.expansion import ExpansionInput, ExpansionOutput
from fastapi import FastAPI
import ray
import warnings

warnings.simplefilter("ignore", SyntaxWarning)

app = FastAPI()


@app.patch("/init-ray", tags=["Ray"])
def init_ray_endpoint():
    return init_ray()


@app.patch("/shutdown-ray", tags=["Ray"])
def shutdown_ray_endpoint():
    return shutdown_ray()


@app.get("/")
def read_root():
    return {"message": "Hello World from Optimization Package!"}


@app.patch("/reconfiguration/bender", tags=["Reconfiguration"])
def reconfiguration_bender(requests: BenderInput) -> ReconfigurationOutput:
    return run_bender(requests)


@app.patch("/reconfiguration/combined", tags=["Reconfiguration"])
def reconfiguration_combined(requests: CombinedInput) -> ReconfigurationOutput:
    return run_combined(requests)


@app.patch("/reconfiguration/admm", tags=["Reconfiguration"])
def reconfiguration_admm(requests: ADMMInput) -> ReconfigurationOutput:
    return run_admm(requests)


@app.patch("/expansion", tags=["Expansion"])
def expansion(
    requests: ExpansionInput,
    with_ray: bool = False,
    cut_file: None | str = None,
    time_now: None | str = None,
) -> ExpansionOutput:
    results = run_expansion(
        requests, with_ray=with_ray, cut_file=cut_file, time_now=time_now
    )
    return results


@app.get("/where-am-i")
def where_am_i_endpoint():
    if not ray.is_initialized():
        init_ray()
    results = ray.get([where_am_i.remote() for _ in range(1000)])
    shutdown_ray()
    return results
