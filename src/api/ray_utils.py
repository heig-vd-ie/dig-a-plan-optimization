import os
import socket
from typing import Any, Dict
import ray


SERVER_RAY_ADDRESS = os.getenv("SERVER_RAY_ADDRESS", None)


@ray.remote
def where_am_i():
    """
    Small Ray utility used to check which node a worker is running on.
    Returns (hostname, node_ip).
    """
    return socket.gethostname(), ray.util.get_node_ip_address()


def init_ray() -> Dict[str, Any]:
    """
    Initialize Ray on a local or remote cluster, depending on SERVER_RAY_ADDRESS.
    Returns basic cluster info (nodes, resources).
    """
    ray.init(
        address=SERVER_RAY_ADDRESS,
        runtime_env={
            "working_dir": os.getcwd(),
            "excludes": ["**/*.parquet", ".cache/outputs**"],
        },
    )
    return {
        "message": "Ray initialized",
        "nodes": ray.nodes(),
        "available_resources": ray.cluster_resources(),
        "used_resources": ray.available_resources(),
    }


def shutdown_ray() -> Dict[str, str]:
    """
    Shutdown Ray cleanly.
    """
    ray.shutdown()
    return {"message": "Ray shutdown"}


def check_ray(with_ray: bool) -> None:
    """
    Log whether Ray is available and initialized given the user's with_ray flag.
    This is intentionally side-effecty (prints), matching the original behavior.
    """
    try:
        import ray

        ray_available = True
    except ImportError:
        ray_available = False

    if ray_available and with_ray and ray.is_initialized():
        print("Running Pipeline with Ray")
    else:
        print("Running Pipeline without Ray")
