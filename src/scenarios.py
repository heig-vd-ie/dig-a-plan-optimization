from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import polars as pl
import patito as pt

from data_schema.load_data import NodeData as NodeLoad  


def generate_random_load_scenarios(
    node_static: pl.DataFrame,
    n_scenarios: int = 50,
    seed: int = 42,
    p_bounds: Tuple[float, float] = (-0.5, 1.0),
    q_bounds: Tuple[float, float] = (-0.5, 1.0),
    zero_slack: bool = True,
) -> Dict[str, pl.DataFrame]:
    """
    Generate randomized p/q load scenarios for every node, validated
    against `data_schema.load_data.NodeData`.

    Parameters
    ----------
    node_static : pl.DataFrame
        The validated static node table (schema: data_schema.node_data.NodeData).
        Must contain at least: ["cn_fk", "node_id", "type"].
    n_scenarios : int
        Number of scenarios to generate.
    seed : int
        RNG seed.
    p_bounds, q_bounds : (float, float)
        Global uniform bounds for active/reactive power (pu).
    zero_slack : bool
        If True, set p=q=0 for nodes whose type == "slack".

    Returns
    -------
    dict[str, pl.DataFrame]
        {"1": df_for_scenario_1, "2": df_for_scenario_2, ...}
        Each DF conforms to `data_schema.load_data.NodeData`.
    """
    rng = np.random.default_rng(seed)

    node_ids = node_static["node_id"].to_numpy()
    cn_fks = node_static["cn_fk"].to_numpy()
    types = node_static["type"].to_list()
    n_nodes = len(node_ids)

    p_min, p_max = p_bounds
    q_min, q_max = q_bounds

    scenarios: Dict[str, pl.DataFrame] = {}

    for i in range(1, n_scenarios + 1):
        p_rand = rng.uniform(low=p_min, high=p_max, size=n_nodes)
        q_rand = rng.uniform(low=q_min, high=q_max, size=n_nodes)

        if zero_slack:
            slack_mask = (pl.Series(types) == "slack").to_numpy()
            p_rand[slack_mask] = 0.0
            q_rand[slack_mask] = 0.0

        df = pl.DataFrame(
            {
                "cn_fk": cn_fks,
                "node_id": node_ids,
                "p_node_pu": p_rand,
                "q_node_pu": q_rand,
                "p_node_max_pu": np.full(n_nodes, p_max),
                "p_node_min_pu": np.full(n_nodes, p_min),
                "q_node_max_pu": np.full(n_nodes, q_max),
                "q_node_min_pu": np.full(n_nodes, q_min),
                "type": types,  
            }
        )

        # Validate against the load schema
        df_pt = (
            pt.DataFrame(df)
            .set_model(NodeLoad)
            .fill_null(strategy="defaults")
            .cast(strict=True)
        )
        df_pt.validate()

        df_valid: pl.DataFrame = df_pt.as_polars() 
        scenarios[str(i)] = df_valid

    return scenarios