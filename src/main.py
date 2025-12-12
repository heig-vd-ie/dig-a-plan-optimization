from api.bender import *
from api.combined import *
from api.admm import *
from api.expansion import *
from api.ray_utils import init_ray, shutdown_ray, where_am_i
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
def reconfiguration_bender(input: BenderInput) -> ReconfigurationOutput:
    return run_bender(input)


@app.patch("/reconfiguration/combined", tags=["Reconfiguration"])
def reconfiguration_combined(input: CombinedInput) -> ReconfigurationOutput:
    return run_combined(input)


@app.patch("/reconfiguration/admm", tags=["Reconfiguration"])
def reconfiguration_admm(input: ADMMInput) -> ReconfigurationOutput:
    return run_admm(input)


@app.patch("/expansion", tags=["Expansion"])
def expansion(
    input: ExpansionInput, with_ray: bool = False, cut_file: None | str = None
) -> ExpansionOutput:
    results = run_expansion(input, with_ray=with_ray, cut_file=cut_file)
    return results


@app.get("/where-am-i")
def where_am_i_endpoint():
    if not ray.is_initialized():
        init_ray()
    results = ray.get([where_am_i.remote() for _ in range(1000)])
    shutdown_ray()
    return results
