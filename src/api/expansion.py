from api import *
from typing import Dict, List


class ExpansionInput(GridCaseModel):
    groups: int | Dict[int, List[int]] = 10


class ExpansionOutput(BaseModel):
    pass


def run_expansion(input: ExpansionInput, with_ray: bool) -> ExpansionOutput:
    _, grid_data = get_grid_case(input)
    expansion_algorithm = ExpansionAlgorithm(
        grid_data=grid_data,
        cache_dir=Path(".cache"),
        admm_groups=input.groups,
        with_ray=with_ray,
    )
    expansion_algorithm.run_pipeline()
    return ExpansionOutput()
