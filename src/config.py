from dataclasses import dataclass
from pathlib import Path
from typing import cast

from dynaconf import Dynaconf


@dataclass
class Settings:
    SWITCH_LINK: str
    SWITCH_PASS: str
    LOAD_ALLOCATION_LOCAL_FOLDER: str


settings_not_casted = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=[".secrets.toml", ".settings.toml"],
    root_path=Path(__file__).parent,
)

settings = cast(
    Settings,
    settings_not_casted,
)
