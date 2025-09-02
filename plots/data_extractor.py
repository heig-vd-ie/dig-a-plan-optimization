import re
import json
import base64
from pymongo import MongoClient


def connect_to_database(host, port, database_name):
    client = MongoClient(host, int(port))
    return client[database_name]


def decode_document_data(data):
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
    return data


def extract_risk_parameters(config):
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


def find_risk_config_in_collection(collection):
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


def get_collections_in_range(db, start_collection=None, end_collection=None):
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
        if start_collection and collection_name < start_collection:
            include = False
        elif end_collection and collection_name > end_collection:
            include = False

        if include:
            matching_collections.append(collection_name)

    return matching_collections


def build_risk_info_from_collection(db, collection_name):
    collection = db[collection_name]
    risk_method, risk_parameter, _ = find_risk_config_in_collection(collection)

    return {
        "collection_name": collection_name,
        "risk_method": risk_method,
        "risk_parameter": risk_parameter,
    }
