from api.bender import *
from fastapi import FastAPI


app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Hello World from Optimization Package!"}


@app.patch("/reconfiguration/bender", tags=["Reconfiguration"])
def reconfiguration_bender(input: BenderInput) -> BenderOutput:
    return run_bender(input)
