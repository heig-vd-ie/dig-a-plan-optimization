from pathlib import Path
from pydantic import BaseModel
from pydantic import BaseModel, Field
from typing import Tuple


class GridCaseModel(BaseModel):
    pp_file: str = Field(
        default="examples/ieee-33/simple_grid.p",
        description="Path to pandapower .p file",
    )
    s_base: float = Field(default=1e6, description="Rated power in Watts")
    cosφ: float = Field(default=0.95, description="Power factor")


class LoadProfiles(BaseModel):
    load_profiles: list[Path] = Field(
        default=[Path("examples/ieee-33/load_profiles")],
        description="List of paths to load profile directories",
    )
    pv_profile: Path = Field(
        default=Path("examples/ieee-33/pv_profiles"),
        description="Path to PV profile directory",
    )
    v_bounds: Tuple[float, float] = Field(
        default=(-0.03, 0.03), description="Voltage bounds in per unit"
    )
    egid_id_mapping_file: Path = Field(
        default=Path("examples/ieee-33/consumer_egid_idx_mapping.csv"),
        description="Mapping of Egid and Node ID",
    )
