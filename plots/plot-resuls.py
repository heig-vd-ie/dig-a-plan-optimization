import os
import re
import json
import base64
import traceback
from pymongo import MongoClient
import plotly.express as px
import pandas as pd

# Configuration
START_COLLECTION = "run_20250902_110438"
END_COLLECTION = "run_20250902_130518"
SERVER_MONGODB_PORT = os.getenv("SERVER_MONGODB_PORT", 27017)


os.chdir(os.getcwd().replace("/src", ""))


def find_risk_config_in_collection(collection):
    """Extract risk method and parameter from collection input document."""
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


def decode_document_data(data):
    """Decode document data if it's encoded."""
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
    """Extract risk method and parameter from config dictionary."""
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


def get_collections_in_range(db, start_collection=None, end_collection=None):
    """Get collections between start and end collection names (inclusive)."""
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
    """Extract risk information from a specific collection."""
    collection = db[collection_name]
    risk_method, risk_parameter, _ = find_risk_config_in_collection(collection)

    return {
        "collection_name": collection_name,
        "risk_method": risk_method,
        "risk_parameter": risk_parameter,
    }


def create_visualization_data(collections_dict, risk_info_dict):
    """Process collections to create visualization data."""
    cursors = {
        name: collection.find({}, {"objectives": 1, "_source_file": 1, "_id": 0})
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
            for obj in doc.get("objectives", []):
                data[name].append(
                    {
                        "objective": obj,
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

    return df_all[
        (df_all["risk_method"] != "Expectation")
        | (df_all["iteration"] == df_all["iteration"].max())
    ]


def create_histogram_plot(df_all):
    """Create normalized histogram plot."""
    fig = px.histogram(
        df_all,
        x="objective",
        color="risk_label",
        nbins=50,
        histnorm="probability density",
        title="Distribution of Objectives by Risk Method",
        labels={
            "objective": "Objective value",
            "count": "Density",
            "risk_label": "Risk Method",
        },
        barmode="group",
    )

    fig.update_layout(
        width=1000,
        height=600,
        legend=dict(title="Risk Method", orientation="v", x=1.02, y=1),
        margin=dict(r=200),
    )

    return fig


def main():
    """Main function to execute the analysis."""
    client = MongoClient("localhost", int(SERVER_MONGODB_PORT))
    db = client["optimization"]

    collections = get_collections_in_range(db, START_COLLECTION, END_COLLECTION)

    all_risk_info = []
    for collection_name in collections:
        risk_info = build_risk_info_from_collection(db, collection_name)
        all_risk_info.append(risk_info)

    if not all_risk_info:
        print("No risk information found!")
        return

    collections_dict = {}
    risk_info_dict = {}

    for info in all_risk_info:
        collection_name = info["collection_name"]
        collections_dict[collection_name] = db[collection_name]
        risk_info_dict[collection_name] = {
            "method": info["risk_method"],
            "parameter": info["risk_parameter"],
        }

    df_all = create_visualization_data(collections_dict, risk_info_dict)
    fig = create_histogram_plot(df_all)
    fig.show()

    summary = (
        df_all.groupby(["risk_method", "risk_parameter"])["objective"]
        .agg(["count", "mean", "std", "min", "max"])
        .round(4)
    )
    print("\nSummary by Risk Method:")
    print(summary)


if __name__ == "__main__":
    main()
