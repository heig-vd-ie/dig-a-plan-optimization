from pathlib import Path
from pydantic import BaseModel
from typing import Dict
from enum import Enum
import json
import os


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


def save_obj_to_json(obj: BaseModel | Dict, path_filename: Path):
    serialized = serialize_obj(obj)

    tmp_path = path_filename.with_suffix(path_filename.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(serialized, f, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp_path, path_filename)


def load_obj_from_json(path_filename: Path) -> Dict:
    return json.load(open(path_filename, "r"))
