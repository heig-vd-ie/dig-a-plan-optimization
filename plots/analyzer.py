from data_extractor import (
    connect_to_database,
    get_collections_in_range,
    build_risk_info_from_collection,
)
from data_processor import create_visualization_data, create_summary_stats


class OptimizationAnalyzer:
    def __init__(self, config):
        self.config = config
        self.db = None
        self.collections = []
        self.risk_info = []
        self.df = None

    def connect(self):
        self.db = connect_to_database(
            self.config.mongodb_host,
            self.config.mongodb_port,
            self.config.database_name,
        )
        return self

    def load_collections(self, start_collection=None, end_collection=None):
        start = start_collection or self.config.start_collection
        end = end_collection or self.config.end_collection

        self.collections = get_collections_in_range(self.db, start, end)
        return self

    def extract_risk_info(self):
        self.risk_info = []
        for collection_name in self.collections:
            risk_info = build_risk_info_from_collection(self.db, collection_name)
            self.risk_info.append(risk_info)
        return self

    def process_data(self):
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
            collection_name = info["collection_name"]
            collections_dict[collection_name] = self.db[collection_name]
            risk_info_dict[collection_name] = {
                "method": info["risk_method"],
                "parameter": info["risk_parameter"],
            }

        self.df = create_visualization_data(collections_dict, risk_info_dict)
        return self

    def get_summary(self):
        if self.df is None:
            raise ValueError("No data available. Run process_data() first.")
        return create_summary_stats(self.df)

    def get_dataframe(self):
        return self.df

    def run_full_analysis(self, start_collection=None, end_collection=None):
        return (
            self.connect()
            .load_collections(start_collection, end_collection)
            .extract_risk_info()
            .process_data()
        )
