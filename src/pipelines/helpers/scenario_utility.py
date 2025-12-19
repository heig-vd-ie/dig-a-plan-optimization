from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import polars as pl
import patito as pt
from data_model import NodeData, LoadData
import random


def generate_random_load_scenarios(
    node_data: pt.DataFrame[NodeData],
    v_slack_node_sqr_pu: float,
    load_data: pl.DataFrame,
    number_of_random_scenarios: int,
    seed: int,
    p_bounds: Tuple[float, float] | None = None,
    q_bounds: Tuple[float, float] | None = None,
    v_bounds: Tuple[float, float] | None = None,
) -> Dict[int, pt.DataFrame[LoadData]]:
    """
    Generate randomized p/q/v load scenarios for every node, validated
    against `data_schema.load_data.NodeData`.
    """
    if p_bounds is None:
        p_bounds = (-0.2, 0.2)
    if q_bounds is None:
        q_bounds = (-0.2, 0.2)
    if v_bounds is None:
        v_bounds = (-0.03, 0.03)

    random.seed(seed)
    rng = np.random.default_rng(seed)

    node_ids = node_data["node_id"].to_numpy()
    n_nodes = len(node_ids)

    p_min, p_max = p_bounds
    q_min, q_max = q_bounds
    v_min, v_max = v_bounds

    scenarios: Dict[int, pt.DataFrame[LoadData]] = {}

    for i in range(1, number_of_random_scenarios + 1):
        random_numbers = rng.uniform(low=0, high=1, size=5 * n_nodes)
        p_rand = random_numbers[0 * n_nodes : 1 * n_nodes] * (p_max - p_min) + p_min
        q_rand = random_numbers[1 * n_nodes : 2 * n_nodes] * (q_max - q_min) + q_min
        pv_rand = random_numbers[2 * n_nodes : 3 * n_nodes] * (p_max - p_min) + p_min
        qv_rand = random_numbers[3 * n_nodes : 4 * n_nodes] * (q_max - q_min) + q_min
        v_rand = random_numbers[4 * n_nodes : 5 * n_nodes] * (v_max - v_min) + v_min

        df = pl.DataFrame(
            {
                "node_id": node_ids,
                "p_cons_pu": abs(
                    (1 + p_rand) * load_data["p_cons_pu"]
                    if i != 1
                    else load_data["p_cons_pu"]
                )
                / (load_data["cons_installed"]),
                "q_cons_pu": (
                    (1 + q_rand) * load_data["q_cons_pu"]
                    if i != 1
                    else load_data["q_cons_pu"]
                )
                / (load_data["cons_installed"]),
                "p_prod_pu": abs(
                    (1 + pv_rand) * load_data["p_prod_pu"]
                    if i != 1
                    else load_data["p_prod_pu"]
                )
                / (load_data["prod_installed"]),
                "q_prod_pu": (
                    (1 + qv_rand) * load_data["q_prod_pu"]
                    if i != 1
                    else load_data["q_prod_pu"]
                )
                / (load_data["prod_installed"]),
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
