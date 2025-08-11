import patito as pt
import random
from data_schema.node_data import NodeData
from pipelines.expansion.models.request import Scenarios, Scenario, Scenarios


def generate_long_term_scenarios(
    nodes: pt.DataFrame[NodeData],
    δ_load_var: float,
    δ_pv_var: float,
    δ_b_var: float,
    number_of_scenarios: int,
    number_of_stages: int,
    seed_number: int = 1234,
) -> Scenarios:
    random.seed(seed_number)
    Ω = []
    for stage in range(number_of_stages):
        Ω.append(
            [
                Scenario(
                    δ_load={
                        str(node["node_id"]): δ_load_var * random.uniform(0, 1)
                        for node in nodes.iter_rows(named=True)
                    },
                    δ_pv={
                        str(node["node_id"]): δ_pv_var * random.uniform(0, 1)
                        for node in nodes.iter_rows(named=True)
                    },
                    δ_b=δ_b_var * random.uniform(0, 1),
                )
                for _ in range(number_of_scenarios)
            ]
        )
    P = [1.0 / number_of_scenarios] * number_of_scenarios
    scenario_data = Scenarios(Ω=Ω, P=P)
    return scenario_data
