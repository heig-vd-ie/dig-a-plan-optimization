from pathlib import Path
from pydantic import BaseModel
from typing import Dict
import json
import os
import ijson
from helpers import generate_log

logger = generate_log(name=__name__)


def serialize_obj(obj: BaseModel) -> Dict:
    return obj.model_dump(mode="json")


def save_obj_to_json(obj: BaseModel | Dict, path_filename: Path):
    try:
        serialized = serialize_obj(obj) if not isinstance(obj, Dict) else obj

        tmp_path = path_filename.with_suffix(path_filename.suffix + ".tmp")

        encoder = json.JSONEncoder(ensure_ascii=False, indent=4)
        with open(tmp_path, "w", encoding="utf-8") as f:

            for chunk in encoder.iterencode(serialized):
                f.write(chunk)

            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_path, path_filename)
    except Exception as err:
        logger.error(f"Error in recording {str(path_filename)} as {err}")


def load_obj_from_json(path_filename: Path) -> Dict:
    """
    Uses ijson to parse the file. Note: ijson.kvitems is excellent
    for large dictionaries to avoid loading the whole tree at once.
    """
    try:
        with open(path_filename, "rb") as f:
            # ijson works best with binary mode ('rb')
            # This reconstructs the dictionary iteratively.
            # For a standard dict, we use ijson.items with an empty prefix ''
            parser = ijson.items(f, "")
            for obj in parser:
                return obj
        return {}
    except Exception as err:
        logger.error(f"Error in loading {str(path_filename)} as {err}")
        return {}
