import json
import polars as pl
from pathlib import Path
from pipeline_reconfiguration import DigAPlanADMM
from helpers.json import load_obj_from_json


def save_dap_state(dap: DigAPlanADMM, base_path=".cache/boisy_dap"):
    """Save DAP state in a way that can be reconstructed for plotting"""
    base_path = Path(base_path)
    base_path.mkdir(exist_ok=True, parents=True)

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
    switch_status = dap.result_manager.extract_switch_status()
    switch_status.write_parquet(base_path / "switch_status.parquet")
    for ω in range(len(dap.model_manager.admm_model_instances)):
        voltage_results = dap.result_manager.extract_node_voltage(ω)
        current_results = dap.result_manager.extract_edge_current(ω)
        voltage_results.write_parquet(base_path / f"voltage_results_{ω}.parquet")
        current_results.write_parquet(base_path / f"current_results_{ω}.parquet")
        for variable_name in [
            "p_curt_cons",
            "p_curt_prod",
            "q_curt_cons",
            "q_curt_prod",
        ]:
            nodal_var = dap.result_manager.extract_nodal_variables(variable_name, ω)
            nodal_var.write_parquet(
                base_path / f"nodal_{variable_name}_results_{ω}.parquet"
            )
        for variable_name in ["p_flow", "q_flow"]:
            edge_var = dap.result_manager.extract_edge_variables(variable_name, ω)
            edge_var.write_parquet(
                base_path / f"edge_{variable_name}_results_{ω}.parquet"
            )
    # Save metadata
    metadata = {
        "konfig": (
            dap.konfig.__dict__ if hasattr(dap.konfig, "__dict__") else str(dap.konfig)
        ),
        "consensus_data": consensus_data,
        "results_data": results_data,
        "time_list": dap.model_manager.time_list,
        "r_norm_list": dap.model_manager.r_norm_list,
        "s_norm_list": dap.model_manager.s_norm_list,
    }

    with open(base_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"DAP state saved to {base_path}")


# Create a mock DAP-like object for plotting
class MockDataManager:
    def __init__(self, path):
        self.node_data = pl.read_parquet(str(path / "node_data.parquet"))
        self.edge_data = pl.read_parquet(str(path / "edge_data.parquet"))


class MockResultManager:
    def __init__(self, path):
        self.base_path = path

    def extract_node_voltage(self, ω):
        return pl.read_parquet(str(self.base_path / f"voltage_results_{ω}.parquet"))

    def extract_edge_current(self, ω):
        return pl.read_parquet(str(self.base_path / f"current_results_{ω}.parquet"))

    def extract_switch_status(self):
        return pl.read_parquet(str(self.base_path / "switch_status.parquet"))

    def extract_nodal_variables(self, variable_name, ω):
        return pl.read_parquet(
            str(self.base_path / f"nodal_{variable_name}_results_{ω}.parquet")
        )

    def extract_edge_variables(self, variable_name, ω):
        return pl.read_parquet(
            str(self.base_path / f"edge_{variable_name}_results_{ω}.parquet")
        )


class MockModelManager:
    def __init__(self, metadata):
        consensus_data = metadata.get("consensus_data", {})
        # Reconstruct consensus variables if they exist
        self.zδ_variable = None
        self.zζ_variable = None
        self.time_list = []  # Dummy placeholder
        self.r_norm_list = []  # Dummy placeholder
        self.s_norm_list = []  # Dummy placeholder
        if consensus_data.get("zδ_variable"):
            self.zδ_variable = pl.from_dict(consensus_data["zδ_variable"])
        if consensus_data.get("zζ_variable"):
            self.zζ_variable = pl.from_dict(consensus_data["zζ_variable"])
        if consensus_data.get("time_list"):
            self.time_list = metadata.get("time_list", [])
        if consensus_data.get("r_norm_list"):
            self.r_norm_list = metadata.get("r_norm_list", [])
        if consensus_data.get("s_norm_list"):
            self.s_norm_list = metadata.get("s_norm_list", [])


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


def load_dap_state(base_path=".cache/boisy_dap"):
    """Load DAP state for plotting purposes"""
    base_path = Path(base_path)

    return MockDigAPlan(base_path)
