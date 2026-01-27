from dataclasses import dataclass
from pathlib import Path
from typing import cast
from dynaconf import Dynaconf


@dataclass
class CacheFolder:
    figures: str
    outputs_expansion: str
    outputs_admm: str
    outputs_bender: str
    outputs_combined: str


@dataclass
class Settings:
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
