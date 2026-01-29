from pathlib import Path
from pydantic import BaseModel
from typing import Dict
from enum import Enum
import json
import gc


def serialize_obj(obj):
    if isinstance(obj, BaseModel):
        obj = obj.model_dump(by_alias=True)
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, list):
        return [serialize_obj(x) for x in obj]
    if isinstance(obj, dict):
        return {k: serialize_obj(v) for k, v in obj.items()}
    return obj


def save_obj_to_json(
    obj: BaseModel | Dict, path_filename: Path, large_file_expected: bool = True
):
    if large_file_expected:
        gc.collect()
    json.dump(
        serialize_obj(obj),
        open(path_filename, "w"),
        indent=4,
        ensure_ascii=False,
    )


def load_obj_from_json(path_filename: Path) -> Dict:
    return json.load(open(path_filename, "r"))
