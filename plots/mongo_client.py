from dataclasses import dataclass
import re
import json
import base64
from typing import Any, Dict, List, Tuple
import pandas as pd
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
from plots import Config


def get_collections_in_range(
    db: Database,
    start_collection: str,
    end_collection: str,
) -> List[str]:
    all_collections = db.list_collection_names()
    run_collections = []
    run_pattern = re.compile(r"^run_(\d{8})_(\d{6})$")

    for collection_name in all_collections:
        match = run_pattern.match(collection_name)
        if match:
            run_collections.append(collection_name)

    run_collections.sort()
    matching_collections = []

    for collection_name in run_collections:
        include = True
        if collection_name < start_collection:
            include = False
        elif collection_name > end_collection:
            include = False

        if include:
            matching_collections.append(collection_name)

    return matching_collections


def connect_to_database(host: str, port: int, database_name: str) -> Database:
    client = MongoClient(host, int(port))
    return client[database_name]


def decode_document_data(data: bytes | str | dict) -> dict:
    if isinstance(data, bytes):
        try:
            decoded_data = data.decode("utf-8")
            return json.loads(decoded_data)
        except Exception:
            try:
                decoded_data = base64.b64decode(data).decode("utf-8")
                return json.loads(decoded_data)
            except Exception:
                return {}
    elif isinstance(data, str):
        try:
            return json.loads(data)
        except Exception:
            return {}
    elif isinstance(data, dict):
        return data
    raise ValueError("Unsupported data type")


def extract_risk_parameters(config: Any) -> Tuple[str | None, str | None]:
    risk_method = None
    risk_parameter = None

    if "expansion" in config and "sddp_params" in config["expansion"]:
        sddp_params = config["expansion"]["sddp_params"]
        risk_method = sddp_params.get("risk_measure_type")
        risk_parameter = sddp_params.get("risk_measure_param")

    if risk_method is None or risk_parameter is None:
        if "sddp_params" in config:
            sddp_params = config["sddp_params"]
            risk_method = risk_method or sddp_params.get("risk_measure_type")
            risk_parameter = risk_parameter or sddp_params.get("risk_measure_param")

        for field in ["risk_method", "risk_type", "method", "approach"]:
            if field in config and risk_method is None:
                risk_method = config[field]
                break

        for field in [
            "risk_parameter",
            "risk_param",
            "epsilon",
            "alpha",
            "beta",
            "parameter",
        ]:
            if field in config and risk_parameter is None:
                risk_parameter = config[field]
                break

    return risk_method, risk_parameter


def find_risk_config_in_collection(
    collection: Collection,
) -> Tuple[str | None, str | None, dict]:
    try:
        input_doc = collection.find_one({"_source_file": "input.json"})
        if input_doc is None:
            input_doc = collection.find_one({"filename": "input.json"})
        if input_doc is None:
            input_doc = collection.find_one({"name": "input.json"})
        if input_doc is None:
            input_doc = collection.find_one(
                {
                    "$or": [
                        {"_source_file": {"$regex": "input"}},
                        {"filename": {"$regex": "input"}},
                        {"name": {"$regex": "input"}},
                    ]
                }
            )

        if input_doc is None:
            return "Unknown", None, {}

        config = input_doc
        if "data" in input_doc:
            config = decode_document_data(input_doc["data"])

        risk_method, risk_parameter = extract_risk_parameters(config)
        return risk_method or "Unknown", risk_parameter, config

    except Exception as e:
        print(f"Error extracting risk info: {e}")
        return "Error", None, {}


@dataclass
class RiskInfo:
    method: str
    parameter: str | None
    config: dict
    collection_name: str


def build_risk_info_from_collection(db: Database, collection_name: str) -> RiskInfo:
    collection = db[collection_name]
    risk_method, risk_parameter, config = find_risk_config_in_collection(collection)

    return RiskInfo(
        method=risk_method or "Unknown",
        parameter=risk_parameter,
        config=config,
        collection_name=collection_name,
    )


class MyMongoClient:
    def __init__(self, config: Config):
        self.config = config
        self.db: Database | None = None
        self.collections: List[str] = []
        self.risk_info: List[RiskInfo] = []
        self.df: pd.DataFrame | None = None

    def connect(self):
        self.db = connect_to_database(
            self.config.mongodb_host,
            self.config.mongodb_port,
            self.config.database_name,
        )
        return self

    def load_collections(self):
        if self.db is None:
            raise ValueError(
                "Database connection not established. Run connect() first."
            )
        self.collections = get_collections_in_range(
            self.db,
            self.config.start_collection,
            self.config.end_collection,
        )
        return self

    def extract_risk_info(self):
        if self.db is None:
            raise ValueError(
                "Database connection not established. Run connect() first."
            )
        self.risk_info = []
        for collection_name in self.collections:
            risk_info = build_risk_info_from_collection(self.db, collection_name)
            self.risk_info.append(risk_info)
        return self

    def create_data(
        self,
        collections_dict: Dict[str, Collection],
        risk_info_dict: Dict[str, Dict[str, Any]],
        field: str,
    ) -> pd.DataFrame:
        cursors = {
            name: collection.find({}, {field: 1, "_source_file": 1, "_id": 0})
            for name, collection in collections_dict.items()
        }

        data = {}
        dfs = {}

        for name, cursor in cursors.items():
            data[name] = []
            risk_method = risk_info_dict[name]["method"]
            risk_param = risk_info_dict[name]["parameter"]

            for doc in cursor:
                file_name = doc.get("_source_file", "unknown")
                for obj in doc.get(field, []):
                    data[name].append(
                        {
                            field: obj,
                            "iteration": int(file_name.split("_")[-1].split(".")[0]),
                            "risk_method": risk_method,
                            "risk_parameter": risk_param,
                            "risk_label": f"{risk_method}"
                            + (f" ({risk_param})" if risk_param is not None else ""),
                        }
                    )

            dfs[name] = pd.DataFrame(data[name])
            dfs[name]["source"] = name

        df_all = pd.concat(dfs.values(), ignore_index=True)

        return df_all

    def process_data(self, field: str):
        if not self.risk_info:
            raise ValueError(
                "No risk information available. Run extract_risk_info() first."
            )

        if self.db is None:
            raise ValueError(
                "Database connection not established. Run connect() first."
            )

        collections_dict = {}
        risk_info_dict = {}

        for info in self.risk_info:
            collection_name = info.collection_name
            collections_dict[collection_name] = self.db[collection_name]
            risk_info_dict[collection_name] = {
                "method": info.method,
                "parameter": info.parameter,
            }

        self.df = self.create_data(collections_dict, risk_info_dict, field)
        return self

    def get_dataframe(self, field: str) -> pd.DataFrame:
        self.connect().load_collections().extract_risk_info().process_data(field)
        if self.df is None:
            raise ValueError("DataFrame is not available. Run process_data() first.")
        return self.df
