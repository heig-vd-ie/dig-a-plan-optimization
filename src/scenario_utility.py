from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import polars as pl
import patito as pt
from data_schema import NodeData, LoadData


def generate_random_load_scenarios(
    node_data: pt.DataFrame[NodeData],
    v_slack_node_sqr_pu: float,
    load_data: pl.DataFrame,
    number_of_random_scenarios: int,
    seed: int = 42,
    p_bounds: Tuple[float, float] = (-0.1, 0.1),
    q_bounds: Tuple[float, float] = (-0.1, 0.1),
    v_bounds: Tuple[float, float] = (-0.03, 0.03),
) -> Dict[int, pt.DataFrame[LoadData]]:
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

    scenarios: Dict[int, pt.DataFrame[LoadData]] = {}

    for i in range(1, number_of_random_scenarios + 1):
        p_rand = rng.uniform(low=p_min, high=p_max, size=n_nodes)
        q_rand = rng.uniform(low=q_min, high=q_max, size=n_nodes)
        pv_rand = rng.uniform(low=p_min, high=p_max, size=n_nodes)
        qv_rand = rng.uniform(low=q_min, high=q_max, size=n_nodes)
        v_rand = rng.uniform(low=v_min, high=v_max, size=n_nodes)

        df = pl.DataFrame(
            {
                "node_id": node_ids,
                "p_cons_pu": (
                    (1 + p_rand) * load_data["p_cons_pu"]
                    if i != 1
                    else load_data["p_cons_pu"]
                ),
                "q_cons_pu": (
                    (1 + q_rand) * load_data["q_cons_pu"]
                    if i != 1
                    else load_data["q_cons_pu"]
                ),
                "p_prod_pu": (
                    (1 + pv_rand) * load_data["p_prod_pu"]
                    if i != 1
                    else load_data["p_prod_pu"]
                ),
                "q_prod_pu": (
                    (1 + qv_rand) * load_data["q_prod_pu"]
                    if i != 1
                    else load_data["q_prod_pu"]
                ),
                "v_node_sqr_pu": (
                    (1 + v_rand) * v_slack_node_sqr_pu
                    if i != 1
                    else v_slack_node_sqr_pu
                ),
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

        scenarios[i] = df_pt

    return scenarios
