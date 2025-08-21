import polars as pl
from pathlib import Path
from pipelines.helpers.json_rw import save_obj_to_json, load_obj_from_json


def save_dap_state(dap, base_path=".cache/boisy_dap"):
    """Save DAP state in a way that can be reconstructed for plotting"""
    base_path = Path(base_path)
    base_path.mkdir(exist_ok=True)

    # Save grid data
    dap.data_manager.node_data.write_parquet(base_path / "node_data.parquet")
    dap.data_manager.edge_data.write_parquet(base_path / "edge_data.parquet")

    # Save consensus results
    consensus_data = {
        "zδ_variable": (
            dap.model_manager.zδ_variable.to_dict()
            if hasattr(dap.model_manager.zδ_variable, "to_dict")
            else None
        ),
        "zζ_variable": (
            dap.model_manager.zζ_variable.to_dict()
            if hasattr(dap.model_manager.zζ_variable, "to_dict")
            else None
        ),
    }

    # Save results if available
    results_data = {}
    try:
        if hasattr(dap, "result_manager") and dap.result_manager:
            voltage_results = dap.result_manager.extract_node_voltage()
            current_results = dap.result_manager.extract_edge_current()
            switch_status = dap.result_manager.extract_switch_status()

            voltage_results.write_parquet(base_path / "voltage_results.parquet")
            current_results.write_parquet(base_path / "current_results.parquet")
            switch_status.write_parquet(base_path / "switch_status.parquet")

            results_data["has_results"] = True
    except:
        results_data["has_results"] = False

    # Save metadata
    metadata = {
        "config": (
            dap.config.__dict__ if hasattr(dap.config, "__dict__") else str(dap.config)
        ),
        "consensus_data": consensus_data,
        "results_data": results_data,
    }
    save_obj_to_json(metadata, base_path / "metadata.json")
    print(f"DAP state saved to {base_path}")


def load_dap_state(base_path=".cache/boisy_dap"):
    """Load DAP state for plotting purposes"""
    base_path = Path(base_path)

    # Create a mock DAP-like object for plotting
    class MockDataManager:
        def __init__(self, path):
            self.node_data = pl.read_parquet(str(path / "node_data.parquet"))
            self.edge_data = pl.read_parquet(str(path / "edge_data.parquet"))

    class MockResultManager:
        def __init__(self, path):
            self.base_path = path

        def extract_node_voltage(self):
            return pl.read_parquet(str(self.base_path / "voltage_results.parquet"))

        def extract_edge_current(self):
            return pl.read_parquet(str(self.base_path / "current_results.parquet"))

        def extract_switch_status(self):
            return pl.read_parquet(str(self.base_path / "switch_status.parquet"))

    class MockModelManager:
        def __init__(self, metadata):
            consensus_data = metadata.get("consensus_data", {})
            # Reconstruct consensus variables if they exist
            self.zδ_variable = None
            self.zζ_variable = None
            if consensus_data.get("zδ_variable"):
                self.zδ_variable = pl.from_dict(consensus_data["zδ_variable"])
            if consensus_data.get("zζ_variable"):
                self.zζ_variable = pl.from_dict(consensus_data["zζ_variable"])

    class MockDigAPlan:
        def __init__(self, path):

            metadata = load_obj_from_json(Path(path / "metadata.json"))

            self.data_manager = MockDataManager(path)
            self.model_manager = MockModelManager(metadata)

            # Only create result manager if results exist
            if metadata.get("results_data", {}).get("has_results", False):
                self.result_manager = MockResultManager(path)
            else:
                self.result_manager = None

    return MockDigAPlan(base_path)
