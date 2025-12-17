from pathlib import Path
from pydantic import BaseModel
from data_model.expansion import ExpansionInput
from data_model.sddp import SDDPResponse, BenderCuts
from datetime import datetime
from pipelines.expansion.algorithm import ExpansionAlgorithm
from pipelines.helpers.json_rw import load_obj_from_json, save_obj_to_json


class DummyObj(BaseModel):
    expansion: ExpansionInput
    time_now: str
    with_ray: bool


def get_session_name() -> str:
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


def run_expansion(
    request: ExpansionInput, with_ray: bool, cut_file: None | str = None
) -> SDDPResponse:
    session_name = get_session_name()
    time_now = session_name
    (Path(".cache/algorithm") / time_now).mkdir(parents=True, exist_ok=True)
    save_obj_to_json(
        DummyObj(expansion=request, time_now=time_now, with_ray=with_ray),
        Path(".cache/algorithm") / time_now / "request.json",
    )
    bender_cuts = (
        None if cut_file is None else BenderCuts(**load_obj_from_json(Path(cut_file)))
    )
    expansion_algorithm = ExpansionAlgorithm(
        request=request, with_ray=with_ray, bender_cuts=bender_cuts, time_now=time_now
    )
    result = expansion_algorithm.run_pipeline()
    return result
