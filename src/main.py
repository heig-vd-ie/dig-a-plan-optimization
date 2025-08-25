from api.bender import *
from api.combined import *
from api.admm import *
from api.expansion import *
from fastapi import FastAPI
import ray

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Hello World from Optimization Package!"}


@app.post("/init-ray")
def init_ray():
    ray.init(
        address="auto",
        runtime_env={
            "working_dir": os.path.dirname(os.path.abspath(__file__)),
        },
    )
    return {
        "message": "Ray initialized",
        "nodes": ray.nodes(),
        "available_resources": ray.cluster_resources(),
        "used_resources": ray.available_resources(),
    }


@app.post("/shutdown-ray")
def shutdown_ray():
    ray.shutdown()
    return {"message": "Ray shutdown"}


@app.patch("/reconfiguration/bender", tags=["Reconfiguration"])
def reconfiguration_bender(input: BenderInput) -> BenderOutput:
    return run_bender(input)


@app.patch("/reconfiguration/combined", tags=["Reconfiguration"])
def reconfiguration_combined(input: CombinedInput) -> CombinedOutput:
    return run_combined(input)


@app.patch("/reconfiguration/admm", tags=["Reconfiguration"])
def reconfiguration_admm(input: ADMMInput) -> ADMMOutput:
    return run_admm(input)


@app.patch("/expansion", tags=["Expansion"])
def expansion(input: ExpansionInput, with_ray: bool = False) -> ExpansionOutput:
    if with_ray and not ray.is_initialized():
        init_ray()
    return run_expansion(input, with_ray=with_ray)


@app.get("/where-am-i")
def where_am_i_endpoint():
    return ray.get([where_am_i.remote() for _ in range(10)])
