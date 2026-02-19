import polars as pl
from polars import col as c
from typing import Optional, Union
import networkx as nx

from helpers.general import generate_log


# Global variable
log = generate_log(name=__name__)


def generate_nx_edge(data: pl.Expr, nx_graph: nx.Graph) -> pl.Expr:
    """
    Generate edges in a NetworkX graph from a Polars expression.

    Args:
        data (pl.Expr): The Polars expression containing edge data.
        nx_graph (nx.Graph): The NetworkX graph.

    Returns:
        pl.Expr: The Polars expression with edges added to the graph.
    """
    return data.map_elements(lambda x: nx_graph.add_edge(**x), return_dtype=pl.Struct)


def get_all_edge_data(nx_graph: nx.Graph) -> pl.DataFrame:
    """
    Get every edge data from a NetworkX graph.

    Args:
        nx_graph (nx.Graph): The NetworkX graph.

    Returns:
        pl.DataFrame: Polar DataFrame containing every edge data.
    """
    return pl.DataFrame(
        list(nx_graph.edges(data=True)),
        strict=False,
        orient="row",
        schema=["u_of_edge", "v_of_edge", "data"],
    ).unnest("data")


def generate_bfs_tree_with_edge_data(graph: nx.Graph, source):
    """
    Create a BFS tree from a graph while retaining edge data.

    Parameters:
        graph (nx.Graph): The input graph.
        source (node): The starting node for BFS.

    Returns:
        nx.DiGraph: A directed BFS tree with edge data preserved.
    """
    # Create an empty directed graph for the BFS tree
    bfs_tree = nx.DiGraph()

    # Add nodes to the BFS tree

    # Add edges to the BFS tree with preserved edge data
    for u, v in nx.bfs_edges(graph, source):

        edge_data = graph.get_edge_data(u, v)
        bfs_tree.add_edge(u, v, **edge_data)

    return bfs_tree


def generate_tree_graph_from_edge_data(
    edge_data: pl.DataFrame,
    slack_node_id: Union[str, int, float],
    data_name: Optional[list[str]] = None,
) -> nx.DiGraph:
    """
    Generate a tree graph from edge data and a specified slack node.

    Args:
        edge_data (pl.DataFrame): The Polars DataFrame containing edge data.
        slack_node_id (Union[str, int, float]): The ID of the slack node.
        data_name (Optional[list[str]], optional): The list of edge data names to include. Defaults to None.

    Returns:
        nx.DiGraph: The generated tree graph with edge data preserved.

    Raises:
        ValueError: If the edge data names are invalid or if the grid is not a connected tree.

    Example:
    >>> import polars as pl
    >>> import networkx as nx
    >>> edge_data = pl.DataFrame({
    ...     "u_of_edge": ["A", "C", "C"],
    ...     "v_of_edge": ["B", "B", "D"],
    ...     "weight": [1, 2, 3]
    ... })
    >>> slack_node_id = "A"
    >>> tree_graph = generate_tree_graph_from_edge_data(edge_data, slack_node_id)
    [("A", "B", {"weight": 1}), ("B", "C", {"weight": 2}), ("C", "D", {"weight": 3})]
    """
    if data_name is None:
        data_selector: pl.Expr = pl.struct(pl.all())
    else:
        if not all(name in edge_data.columns for name in data_name):
            raise ValueError("Invalid edge data name")
        if not all(name in data_name for name in ["u_of_edge", "v_of_edge"]):
            raise ValueError("Missing u_of_edge or v_of_edge")

        data_selector: pl.Expr = pl.struct(data_name)

    if edge_data.filter(
        pl.any_horizontal(c("u_of_edge", "v_of_edge") == slack_node_id)
    ).is_empty():
        raise ValueError("The slack node is not in the grid")

    nx_grid: nx.Graph = nx.Graph()
    _ = edge_data.with_columns(data_selector.pipe(generate_nx_edge, nx_graph=nx_grid))

    if not nx.is_tree(nx_grid):
        raise ValueError("The grid is not a tree")
    elif not nx.is_connected(nx_grid):
        raise ValueError("The grid is not connected")

    return generate_bfs_tree_with_edge_data(nx_grid, slack_node_id)
