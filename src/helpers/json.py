from pathlib import Path
from pydantic import BaseModel
from typing import Dict
from enum import Enum
import json
import os
import ijson


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

    encoder = json.JSONEncoder(ensure_ascii=False, indent=4)
    with open(tmp_path, "w", encoding="utf-8") as f:

        for chunk in encoder.iterencode(serialized):
            f.write(chunk)

        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp_path, path_filename)


def load_obj_from_json(path_filename: Path) -> Dict:
    """
    Uses ijson to parse the file. Note: ijson.kvitems is excellent
    for large dictionaries to avoid loading the whole tree at once.
    """
    with open(path_filename, "rb") as f:
        # ijson works best with binary mode ('rb')
        # This reconstructs the dictionary iteratively.
        # For a standard dict, we use ijson.items with an empty prefix ''
        parser = ijson.items(f, "")
        for obj in parser:
            return obj
    return {}
