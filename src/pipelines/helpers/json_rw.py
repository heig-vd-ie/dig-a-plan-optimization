from pathlib import Path
from pydantic import BaseModel
from typing import Dict
import json


def save_obj_to_json(obj: BaseModel | Dict, path_filename: Path):
    json.dump(
        obj.model_dump(by_alias=True) if isinstance(obj, BaseModel) else obj,
        open(path_filename, "w"),
        indent=4,
        ensure_ascii=False,
    )


def load_obj_from_json(path_filename: Path) -> Dict:
    return json.load(open(path_filename, "r"))
