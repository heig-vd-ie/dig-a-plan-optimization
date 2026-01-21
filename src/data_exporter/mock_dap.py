import json
import re
import polars as pl
from pathlib import Path
from typing import Protocol, runtime_checkable

from pipeline_reconfiguration import DigAPlanADMM, DigAPlanBender, DigAPlanCombined
from helpers.json import load_obj_from_json

def _write_parquet_any(obj, path: Path) -> None:
    """Write parquet for polars DataFrame/LazyFrame or pandas DataFrame."""
    if isinstance(obj, pl.DataFrame):
        obj.write_parquet(path)
        return
    if isinstance(obj, pl.LazyFrame):
        obj.collect().write_parquet(path)
        return
    if hasattr(obj, "to_parquet"):  # pandas DataFrame
        obj.to_parquet(path)
        return
    raise TypeError(f"Cannot write parquet for type {type(obj)} to {path}")


# =============================================================================
# Factory Pattern: Savers
# =============================================================================

@runtime_checkable
class DapStateSaver(Protocol):
    kind: str
    def save(self, dap, base_path: Path) -> None:
        ...

class BaseStateSaver:
    def save_common(self, dap: DigAPlanADMM | DigAPlanBender | DigAPlanCombined, base_path: Path) -> None:
        base_path.mkdir(exist_ok=True, parents=True)

        # Save grid data 
        _write_parquet_any(dap.data_manager.node_data, base_path / "node_data.parquet")
        _write_parquet_any(dap.data_manager.edge_data, base_path / "edge_data.parquet")

        # Save switch status 
        switch_status = dap.result_manager.extract_switch_status()
        _write_parquet_any(switch_status, base_path / "switch_status.parquet")
        
class ADMMStateSaver(BaseStateSaver):
    kind = "admm"

    def save(self, dap: DigAPlanADMM, base_path: Path) -> None:
        self.save_common(dap, base_path)

        consensus_data = {
            "zδ_variable": dap.model_manager.zδ_variable,
            "zζ_variable": dap.model_manager.zζ_variable,
        }

        for ω in range(len(dap.model_manager.admm_model_instances)):
            voltage_results = dap.result_manager.extract_node_voltage(ω)
            current_results = dap.result_manager.extract_edge_current(ω)
            _write_parquet_any(voltage_results, base_path / f"voltage_results_{ω}.parquet")
            _write_parquet_any(current_results, base_path / f"current_results_{ω}.parquet")

            for variable_name in ["p_curt_cons", "p_curt_prod", "q_curt_cons", "q_curt_prod"]:
                nodal_var = dap.result_manager.extract_nodal_variables(variable_name, ω)
                _write_parquet_any(nodal_var, base_path / f"nodal_{variable_name}_results_{ω}.parquet")

            for variable_name in ["p_flow", "q_flow"]:
                edge_var = dap.result_manager.extract_edge_variables(variable_name, ω)
                _write_parquet_any(edge_var, base_path / f"edge_{variable_name}_results_{ω}.parquet")

        metadata = {
            "kind": "admm",
            "konfig": (dap.konfig.__dict__ if hasattr(dap.konfig, "__dict__") else str(dap.konfig)),
            "time_list": dap.model_manager.time_list,
            "r_norm_list": dap.model_manager.r_norm_list,
            "s_norm_list": dap.model_manager.s_norm_list,
            "scenarios": list(dap.model_manager.admm_model_instances.keys()),
        }

        with open(base_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        with open(base_path / "consensus_variables_zdelta.parquet", "wb") as f:
            consensus_data["zδ_variable"].write_parquet(f)

        with open(base_path / "consensus_variables_zzeta.parquet", "wb") as f:
            consensus_data["zζ_variable"].write_parquet(f)

        print(f"DAP state saved to {base_path}")

class BenderStateSaver(BaseStateSaver):
    kind = "bender"

    def save(self, dap: DigAPlanBender, base_path: Path) -> None:
        self.save_common(dap, base_path)

        voltage_results = dap.result_manager.extract_node_voltage()
        current_results = dap.result_manager.extract_edge_current()
        tap_position = dap.result_manager.extract_transformer_tap_position()

        _write_parquet_any(voltage_results, base_path / "voltage_results_0.parquet")
        _write_parquet_any(current_results, base_path / "current_results_0.parquet")
        _write_parquet_any(tap_position, base_path / "tap_position.parquet")

        if hasattr(dap.result_manager, "extract_edge_variables"):
            for variable_name in ["p_flow", "q_flow"]:
                edge_var = dap.result_manager.extract_edge_variables(variable_name)
                _write_parquet_any(edge_var, base_path / f"edge_{variable_name}_results_0.parquet")

        metadata = {
            "kind": "bender",
            "konfig": (dap.konfig.__dict__ if hasattr(dap.konfig, "__dict__") else str(dap.konfig)),
            "slave_obj_list": getattr(dap.model_manager, "slave_obj_list", []),
            "master_obj_list": getattr(dap.model_manager, "master_obj_list", []),
            "convergence_list": getattr(dap.model_manager, "convergence_list", []),
        }

        with open(base_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        print(f"DAP state saved to {base_path}")
        
class CombinedStateSaver(BaseStateSaver):
    kind = "combined"

    def save(self, dap: DigAPlanCombined, base_path: Path) -> None:
        self.save_common(dap, base_path)

        voltage_results = dap.result_manager.extract_node_voltage()
        current_results = dap.result_manager.extract_edge_current()
        tap_position = dap.result_manager.extract_transformer_tap_position()

        _write_parquet_any(voltage_results, base_path / "voltage_results_0.parquet")
        _write_parquet_any(current_results, base_path / "current_results_0.parquet")
        _write_parquet_any(tap_position, base_path / "tap_position.parquet")

        # save p_flow/q_flow 
        if hasattr(dap.result_manager, "extract_edge_variables"):
            for variable_name in ["p_flow", "q_flow"]:
                edge_var = dap.result_manager.extract_edge_variables(variable_name)
                _write_parquet_any(edge_var, base_path / f"edge_{variable_name}_results_0.parquet")

        metadata = {
            "kind": "combined",
            "konfig": (
                dap.konfig.__dict__ if hasattr(dap.konfig, "__dict__") else str(dap.konfig)
            ),
            "convergence_list": getattr(dap.model_manager, "convergence_list", []),
        }

        with open(base_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)


class DapStateSaverFactory:
    """Factory using a registry mapping."""

    _registry = {
        DigAPlanADMM: ADMMStateSaver,
        DigAPlanBender: BenderStateSaver,
        DigAPlanCombined: CombinedStateSaver,
    }

    @staticmethod
    def create(dap) -> DapStateSaver:
        for dap_type, saver_cls in DapStateSaverFactory._registry.items():
            if isinstance(dap, dap_type):
                return saver_cls()
        raise TypeError(f"Unsupported DAP pipeline type: {type(dap)}")

def save_dap_state(dap: DigAPlanADMM | DigAPlanBender | DigAPlanCombined, base_path=".cache/boisy_dap"):
    """Save DAP state using factory-selected saver."""
    base_path = Path(base_path)
    saver = DapStateSaverFactory.create(dap)
    saver.save(dap, base_path)

# =============================================================================
# Mock objects for plotting (loader)
# =============================================================================
class MockDataManager:
    def __init__(self, path: Path):
        self.node_data = pl.read_parquet(str(path / "node_data.parquet"))
        self.edge_data = pl.read_parquet(str(path / "edge_data.parquet"))


class MockResultManager:
    def __init__(self, path: Path):
        self.base_path = path

    def extract_node_voltage(self, ω=0):
        return pl.read_parquet(str(self.base_path / f"voltage_results_{ω}.parquet"))

    def extract_edge_current(self, ω=0):
        return pl.read_parquet(str(self.base_path / f"current_results_{ω}.parquet"))

    def extract_switch_status(self):
        return pl.read_parquet(str(self.base_path / "switch_status.parquet"))
    
    def extract_transformer_tap_position(self):
        return pl.read_parquet(str(self.base_path / "tap_position.parquet"))

    def extract_nodal_variables(self, variable_name, ω=0):
        return pl.read_parquet(
            str(self.base_path / f"nodal_{variable_name}_results_{ω}.parquet")
        )

    def extract_edge_variables(self, variable_name, ω=0):
        return pl.read_parquet(
            str(self.base_path / f"edge_{variable_name}_results_{ω}.parquet")
        )


def parse_series_string(s, cast=str):
    body = re.search(r"(.∗)(.∗)", s, flags=re.S)
    if not body:
        return []
    lines = body.group(1).splitlines()
    values = []
    for line in lines:
        line = line.strip()
        if not line or line == "…":
            continue
        line = line.strip('"')
        values.append(cast(line))
    return values


class MockModelManager:
    def __init__(self, path: Path):
        metadata = load_obj_from_json(Path(path / "metadata.json"))
        kind = metadata.get("kind", "admm")
        
        # Common defaults
        self.zδ_variable = None
        self.zζ_variable = None
        
        
        self.time_list = []
        self.r_norm_list = []
        self.s_norm_list = []

        self.slave_obj_list = []
        self.master_obj_list = []
        self.convergence_list = []
        
        if kind == "admm":
            
            consensus_data_zdelta = pl.read_parquet(
                Path(path / "consensus_variables_zdelta.parquet")
            )
            consensus_data_zzeta = pl.read_parquet(
                Path(path / "consensus_variables_zzeta.parquet")
            )
            # Reconstruct consensus variables 
            self.zδ_variable = consensus_data_zdelta
            self.zζ_variable = consensus_data_zzeta
            self.time_list = metadata.get("time_list", [])
            self.r_norm_list = metadata.get("r_norm_list", [])
            self.s_norm_list = metadata.get("s_norm_list", [])
            self.admm_model_instances = {ω: None for ω in metadata.get("scenarios", [])}
            
            return
        if kind == "bender":
            self.slave_obj_list = metadata.get("slave_obj_list", [])
            self.master_obj_list = metadata.get("master_obj_list", [])
            self.convergence_list = metadata.get("convergence_list", [])
            return 
        
        if kind == "combined":
            self.convergence_list = metadata.get("convergence_list", [])
            return     
        


class MockDigAPlan:
    def __init__(self, path):
        self.data_manager = MockDataManager(path)
        self.model_manager = MockModelManager(path)
        self.result_manager = MockResultManager(path)


def load_dap_state(base_path=".cache/boisy_dap"):
    """Load DAP state for plotting purposes"""
    base_path = Path(base_path)

    return MockDigAPlan(base_path)
