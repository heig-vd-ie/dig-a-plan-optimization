from api.bender import *
from api.combined import *
from api.admm import *
from fastapi import FastAPI


app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Hello World from Optimization Package!"}


@app.patch("/reconfiguration/bender", tags=["Reconfiguration"])
def reconfiguration_bender(input: BenderInput) -> BenderOutput:
    return run_bender(input)


@app.patch("/reconfiguration/combined", tags=["Reconfiguration"])
def reconfiguration_combined(input: CombinedInput) -> CombinedOutput:
    return run_combined(input)


@app.patch("/reconfiguration/admm", tags=["Reconfiguration"])
def reconfiguration_admm(input: ADMMInput) -> ADMMOutput:
    return run_admm(input)
