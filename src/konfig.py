from dataclasses import dataclass
from pathlib import Path
from typing import cast, Dict
from dynaconf import Dynaconf


@dataclass
class Kace:
    pandapower_file: str
    geojson_file: str
    load_allocation_folder: str
    load_gpkg_file: str
    load_duckdb_file: str
    dap_state_file: str
    dap_state_fixed_file: str
    dump_net_file: str


@dataclass
class CacheFolder:
    figures: str
    outputs_expansion: str
    outputs_admm: str
    outputs_bender: str
    outputs_combined: str


@dataclass
class Settings:
    cases: Dict[str, Kace]
    cache: CacheFolder


settings_not_casted = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=[".secrets.toml", ".settings.toml"],
    root_path=Path(__file__).parent,
)

settings = cast(
    Settings,
    settings_not_casted,
)
