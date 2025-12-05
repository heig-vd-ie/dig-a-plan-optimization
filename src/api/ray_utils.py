import socket

import ray


@ray.remote
def where_am_i():
    """
    Small Ray utility used to check which node a worker is running on.
    Returns (hostname, node_ip).
    """
    return socket.gethostname(), ray.util.get_node_ip_address()


__all__ = ["where_am_i"]
