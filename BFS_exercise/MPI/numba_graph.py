from numba import njit
import numpy as np
from numba.typed import List as NumbaList

@njit(parallel=True)
def generate_graph_numba(num_nodes: int, connectivity_pct: float = 0.5):    # memory complexity = (num_nodes)^2 * connectivity_pct
    """
    Generates a random directed graph as a Numba-compatible adjacency list:
    - A NumbaList of NumbaLists containing neighbor indices.

    Parameters:
        num_nodes (int): Number of nodes.
        connectivity_pct (float): percentage of num_nodes each node will have as neighbors

    Returns:
        adj_list (NumbaList[NumbaList[int]]): Numba-compatible adjacency list.
    """
    np.random.seed(42)

    # use for picking which nodes to connect as neighbors
    node_ids = np.arange(num_nodes)
    neighbors_per_node = int(num_nodes * connectivity_pct) # rounded to int
    
    # Initialize empty adjacency list for all nodes
    adj_list = NumbaList()

    # give all nodes x % connections to all other nodes
    # use index as node id (index 0 is node 0, etc.)
    for _ in range(num_nodes):  # loop for all parent nodes
        neighbors: np.ndarray = np.random.choice(node_ids, size=neighbors_per_node, replace=False)  # dont care if node is its own neighbor. BFS uses `visited` to avoid the self-loop
        neighbor_list = NumbaList(neighbors)
        adj_list.append(neighbor_list)

    return adj_list