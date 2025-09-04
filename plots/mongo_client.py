from dataclasses import dataclass
import re
import json
import base64
from typing import Any, Dict, List, Tuple, Optional, Callable, Union
import pandas as pd
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
from pymongo.cursor import Cursor


@dataclass
class MongoConfig:
    mongodb_port: int = 27017
    mongodb_host: str = "localhost"
    database_name: str = "optimization"
    histogram_bins: int = 50
    plot_width: int = 1000
    plot_height: int = 600
    start_collection: str = ""
    end_collection: str = ""


@dataclass
class FieldExtractor:
    field_path: str
    output_name: Optional[str] = None
    transform_func: Optional[Callable[[Any], Any]] = None

    def __post_init__(self):
        if self.output_name is None:
            self.output_name = self.field_path.replace(".", "_")


@dataclass
class CursorConfig:
    filter_query: Dict[str, Any]
    projection: Optional[Dict[str, int]] = None
    sort: Optional[List[Tuple[str, int]]] = None
    limit: Optional[int] = None


def extract_nested_field(doc: Dict[str, Any], field_path: str) -> Any:
    keys = field_path.split(".")
    value = doc

    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return None

    return value


def decode_document_data(data: Union[bytes, str, dict]) -> dict:
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


def connect_to_database(host: str, port: int, database_name: str) -> Database:
    client = MongoClient(host, int(port))
    return client[database_name]


def extract_file_metadata(filename: str) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {"filename": filename}

    # Extract iteration number from sddp_response pattern
    iteration_match = re.search(r"sddp_response_(\d+)", filename)
    if iteration_match:
        metadata["iteration"] = int(iteration_match.group(1))
        metadata["file_type"] = "sddp_response"

    # Extract ADMM iteration, stage, scenario
    admm_match = re.search(r"admm_result_iter(\d+)_stage(\d+)_scen(\d+)", filename)
    if admm_match:
        metadata["iteration"] = int(admm_match.group(1))
        metadata["stage"] = int(admm_match.group(2))
        metadata["scenario"] = int(admm_match.group(3))
        metadata["file_type"] = "admm_result"

    # Extract timestamp patterns
    timestamp_match = re.search(r"(\d{8}_\d{6})", filename)
    if timestamp_match:
        metadata["timestamp"] = timestamp_match.group(1)

    return metadata


class GeneralMongoClient:
    def __init__(self, config: MongoConfig):
        self.config = config
        self.db: Optional[Database] = None
        self.collections: List[str] = []

    def connect(self) -> "GeneralMongoClient":
        self.db = connect_to_database(
            self.config.mongodb_host,
            self.config.mongodb_port,
            self.config.database_name,
        )
        return self

    def load_collections(self) -> "GeneralMongoClient":
        if self.db is None:
            raise ValueError(
                "Database connection not established. Run connect() first."
            )

        all_collections = self.db.list_collection_names()

        all_collections.sort()
        self.collections = [
            name
            for name in all_collections
            if self.config.start_collection <= name <= self.config.end_collection
        ]

        return self

    def extract_data_with_cursors(
        self,
        cursor_configs: Dict[str, CursorConfig],
        field_extractors: List[FieldExtractor],
        collection_subset: Optional[List[str]] = None,
        include_metadata: bool = True,
    ) -> pd.DataFrame:
        if self.db is None:
            raise ValueError(
                "Database connection not established. Run connect() first."
            )

        target_collections = collection_subset or self.collections
        all_data = []

        for collection_name in target_collections:
            collection = self.db[collection_name]

            cursor_config = cursor_configs.get(
                collection_name, CursorConfig(filter_query={})
            )

            cursor = collection.find(
                cursor_config.filter_query, cursor_config.projection
            )

            if cursor_config.sort:
                cursor = cursor.sort(cursor_config.sort)
            if cursor_config.limit:
                cursor = cursor.limit(cursor_config.limit)

            for doc in cursor:
                base_row: Dict[str, Any] = {"collection": collection_name}

                if include_metadata:
                    source_file = doc.get(
                        "_source_file", doc.get("filename", "unknown")
                    )
                    file_metadata = extract_file_metadata(source_file)
                    base_row.update(file_metadata)

                # Extract requested fields
                extracted_data = {}
                for extractor in field_extractors:
                    value = extract_nested_field(doc, extractor.field_path)

                    if extractor.transform_func and value is not None:
                        value = extractor.transform_func(value)

                    extracted_data[extractor.output_name] = value

                if any(isinstance(extracted_data[key], list) for key in extracted_data):
                    # If any field is a list, create multiple rows
                    max_length = max(
                        len(val) if isinstance(val, list) else 1
                        for val in extracted_data.values()
                    )

                    for i in range(max_length):
                        row: Dict[str, Any] = base_row.copy()
                        for key, value in extracted_data.items():
                            if isinstance(value, list) and i < len(value):
                                row[key] = value[i]
                            elif not isinstance(value, list):
                                row[key] = value
                            else:
                                row[key] = None
                        row["list_index"] = i
                        all_data.append(row)
                else:
                    # Single row
                    row: Dict[str, Any] = base_row.copy()
                    row.update(extracted_data)
                    all_data.append(row)

        return pd.DataFrame(all_data)

    def extract_simulations_data(
        self,
        cursor_configs: Dict[str, CursorConfig],
        collection_subset: Optional[List[str]] = None,
        include_metadata: bool = True,
    ) -> pd.DataFrame:
        if self.db is None:
            raise ValueError(
                "Database connection not established. Run connect() first."
            )

        target_collections = collection_subset or self.collections
        all_data = []

        for collection_name in target_collections:
            collection = self.db[collection_name]

            cursor_config = cursor_configs.get(
                collection_name, CursorConfig(filter_query={})
            )
            cursor = collection.find(
                cursor_config.filter_query, cursor_config.projection
            )

            if cursor_config.sort:
                cursor = cursor.sort(cursor_config.sort)
            if cursor_config.limit:
                cursor = cursor.limit(cursor_config.limit)

            for doc in cursor:
                base_row: Dict[str, Any] = {"collection": collection_name}

                if include_metadata:
                    source_file = doc.get(
                        "_source_file", doc.get("filename", "unknown")
                    )
                    file_metadata = extract_file_metadata(source_file)
                    base_row.update(file_metadata)

                # Handle simulations data structure
                simulations = doc.get("simulations", [])
                for sim_idx, simulation in enumerate(simulations):
                    if isinstance(simulation, list):
                        for stage_idx, stage in enumerate(simulation):
                            if isinstance(stage, dict):
                                row: Dict[str, Any] = base_row.copy()
                                row.update(
                                    {
                                        "simulation": sim_idx,
                                        "stage": stage_idx,
                                        **stage,
                                    }
                                )
                                all_data.append(row)

        return pd.DataFrame(all_data)

    def get_sample_document(self, collection_name: str, limit: int = 1) -> List[Dict]:
        if self.db is None:
            raise ValueError(
                "Database connection not established. Run connect() first."
            )

        collection = self.db[collection_name]
        return list(collection.find({}).limit(limit))


class MyMongoClient(GeneralMongoClient):
    def __init__(self, config: MongoConfig):
        super().__init__(config)
        self.risk_info: List[Any] = []
        self.df: Optional[pd.DataFrame] = None

    def extract_risk_info(self) -> "MyMongoClient":
        if self.db is None:
            raise ValueError(
                "Database connection not established. Run connect() first."
            )

        self.risk_info = []

        for collection_name in self.collections:
            collection = self.db[collection_name]

            for doc in collection.find(
                {
                    "$or": [
                        {"additional_params": {"$exists": True}},
                        {"planning_params": {"$exists": True}},
                        {"_source_file": {"$regex": "input|config|request"}},
                    ]
                }
            ):
                risk_data: Dict[str, Any] = {"collection": collection_name}

                if "additional_params" in doc:
                    params = doc["additional_params"]
                    if isinstance(params, dict):
                        risk_data["risk_measure_type"] = params.get("risk_measure_type")
                        risk_data["risk_measure_param"] = params.get(
                            "risk_measure_param"
                        )
                        risk_data["n_simulations"] = params.get("n_simulations")
                        risk_data["iteration_limit"] = params.get("iteration_limit")
                        risk_data["seed"] = params.get("seed")

                if "planning_params" in doc:
                    params = doc["planning_params"]
                    if isinstance(params, dict):
                        risk_data["planning_risk_type"] = params.get(
                            "risk_measure_type"
                        )
                        risk_data["planning_risk_param"] = params.get(
                            "risk_measure_param"
                        )

                risk_data["source_file"] = doc.get("_source_file", "unknown")

                if any(key.startswith("risk_") for key in risk_data.keys()):
                    self.risk_info.append(risk_data)

        return self

    def _extract_objectives(self) -> pd.DataFrame:
        cursor_configs = {
            collection_name: CursorConfig(
                filter_query={"_source_file": {"$regex": "sddp_response"}},
                projection={"objectives": 1, "_source_file": 1, "_id": 0},
            )
            for collection_name in self.collections
        }

        field_extractors = [
            FieldExtractor(field_path="objectives", output_name="objective_value")
        ]

        return self.extract_data_with_cursors(
            cursor_configs=cursor_configs,
            field_extractors=field_extractors,
            include_metadata=True,
        )

    def extract_objectives(self) -> pd.DataFrame:

        objectives_df = self._extract_objectives()
        self.extract_risk_info()
        risk_df = pd.DataFrame(self.risk_info)

        risk_mapping = risk_df.set_index("collection")[
            ["risk_measure_type", "risk_measure_param"]
        ].to_dict("index")

        objectives_df = objectives_df.copy()
        objectives_df["risk_method"] = objectives_df["collection"].map(
            lambda x: risk_mapping.get(x, {}).get("risk_measure_type", "Unknown")
        )
        objectives_df["risk_param"] = objectives_df["collection"].map(
            lambda x: risk_mapping.get(x, {}).get("risk_measure_param", None)
        )
        objectives_df["risk_label"] = objectives_df.apply(
            lambda row: (
                f"{row['risk_method']} (α={row['risk_param']})"
                if row["risk_param"] is not None
                else row["risk_method"]
            ),
            axis=1,
        )

        return objectives_df

    def _extract_simulations(self) -> pd.DataFrame:
        cursor_configs = {
            collection_name: CursorConfig(
                filter_query={"_source_file": {"$regex": "sddp_response"}},
                projection={"simulations": 1, "_source_file": 1, "_id": 0},
            )
            for collection_name in self.collections
        }

        return self.extract_simulations_data(
            cursor_configs=cursor_configs,
            include_metadata=True,
        )

    def extract_simulations(self) -> pd.DataFrame:
        simulations_df = self._extract_simulations()
        self.extract_risk_info()
        risk_df = pd.DataFrame(self.risk_info)

        risk_mapping = risk_df.set_index("collection")[
            ["risk_measure_type", "risk_measure_param"]
        ].to_dict("index")

        simulations_df = simulations_df.copy()
        simulations_df["risk_method"] = simulations_df["collection"].map(
            lambda x: risk_mapping.get(x, {}).get("risk_measure_type", "Unknown")
        )
        simulations_df["risk_param"] = simulations_df["collection"].map(
            lambda x: risk_mapping.get(x, {}).get("risk_measure_param", None)
        )
        simulations_df["risk_label"] = simulations_df.apply(
            lambda row: (
                f"{row['risk_method']} (α={row['risk_param']})"
                if row["risk_param"] is not None
                else row["risk_method"]
            ),
            axis=1,
        )

        return simulations_df
