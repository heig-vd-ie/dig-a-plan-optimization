from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import polars as pl
import patito as pt
from data_schema.node_data import NodeData
from data_schema.load_data import LoadData


def generate_random_load_scenarios(
    node_data: pt.DataFrame[NodeData],
    n_scenarios: int = 50,
    seed: int = 42,
    p_bounds: Tuple[float, float] = (-0.5, 1.0),
    q_bounds: Tuple[float, float] = (-0.5, 1.0),
    v_bounds: Tuple[float, float] = (0.9, 1.1),
) -> Dict[str, pt.DataFrame[LoadData]]:
    """
    Generate randomized p/q/v load scenarios for every node, validated
    against `data_schema.load_data.NodeData`.
    """
    rng = np.random.default_rng(seed)

    node_ids = node_data["node_id"].to_numpy()
    n_nodes = len(node_ids)

    p_min, p_max = p_bounds
    q_min, q_max = q_bounds
    v_min, v_max = v_bounds

    scenarios: Dict[str, pt.DataFrame[LoadData]] = {}

    for i in range(1, n_scenarios + 1):
        p_rand = rng.uniform(low=p_min, high=p_max, size=n_nodes)
        q_rand = rng.uniform(low=q_min, high=q_max, size=n_nodes)
        v_rand = rng.uniform(low=v_min, high=v_max, size=n_nodes)

        df = pl.DataFrame(
            {
                "node_id": node_ids,
                "p_node_pu": p_rand,
                "q_node_pu": q_rand,
                "v_node_sqr_pu": v_rand**2,
            }
        )

        # Validate against the load schema
        df_pt = (
            pt.DataFrame(df)
            .set_model(LoadData)
            .fill_null(strategy="defaults")
            .cast(strict=True)
        )
        df_pt.validate()

        scenarios[str(i)] = df_pt

    return scenarios
