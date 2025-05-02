"""
Compares a parallel and sequential Numba Jit'ed version of BFS
That means that the comparison is between 2 compiled approaches
"""

from numba import njit, prange      # njit = jit(nopython=True),  prange = parallel range
import numpy as np
import time
from numba.typed import List as NumbaList


# -----------------------------------------------
# Timer decorator for benchmarking
# -----------------------------------------------
def timer(f):
    def wrapper(*args, **kwargs):
        t0 = time.time()
        result = f(*args, **kwargs)
        t1 = time.time()
        return result, (t1 - t0)
    return wrapper


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


# -----------------------------------------------
# Numba-parallel BFS using adjacency list format
# -----------------------------------------------
@timer
@njit(parallel=True)
def bfs_parallel(adj_list, start_node_id, n):
    """
    Parallel BFS that directly operates on an adjacency list (List[List[int]]).
    Each BFS level is expanded in parallel using prange.
    """
    visited = np.zeros(n, dtype=np.uint8)       # starts as all 0's to indicate unvisited state
    dist = -np.ones(n, dtype=np.int32)          # starts as all -1's. If when done any distances are still -1, node was unreachable from start node

    visited[start_node_id] = 1
    dist[start_node_id] = 0
    current_frontier = [start_node_id]

    while len(current_frontier) > 0:
        next_frontier_flags = np.zeros(n, dtype=np.uint8)
        next_distances = -np.ones(n, dtype=np.int32)

        # Parallel loop over the current frontier
        for i in prange(len(current_frontier)):
            node = current_frontier[i]
            neighbors = adj_list[node]

            for j in range(len(neighbors)):
                neighbor = neighbors[j]
                if visited[neighbor] == 0:                      # if neighbor not visited yet 
                    next_frontier_flags[neighbor] = 1           # neighbor is to be visited on next frontier
                    next_distances[neighbor] = dist[node] + 1   # neighbor is this distance from source through current node

        # Sequentially apply the updates to global arrays
        new_frontier = []
        for i in range(n):
            # Check for race conditions (if a node is to be visited now, but was already visited)
            if next_frontier_flags[i] == 1 and visited[i] == 0:
                visited[i] = 1
                dist[i] = next_distances[i]
                new_frontier.append(i)

        current_frontier = new_frontier

    return dist


# -----------------------------------------------
# Numba-sequential BFS using adjacency list format
# -----------------------------------------------
@timer
@njit()
def bfs_sequential(adj_list, start_node_id, n):
    """
    Sequential BFS that directly operates on an adjacency list (List[List[int]]).
    Each BFS level is expanded in sequential using prange.

    There are probably some things that could be more sequentially efficient, but this code was taken from the parallel
    version and simply made to be sequential while still using JIT.
    """
    visited = np.zeros(n, dtype=np.uint8)       # starts as all 0's to indicate unvisited state
    dist = -np.ones(n, dtype=np.int32)          # starts as all -1's. If when done any distances are still -1, node was unreachable from start node

    visited[start_node_id] = 1
    dist[start_node_id] = 0
    current_frontier = [start_node_id]

    while len(current_frontier) > 0:
        next_frontier_flags = np.zeros(n, dtype=np.uint8)
        next_distances = -np.ones(n, dtype=np.int32)

        # Now also sequentially loop over the current frontier
        for i in range(len(current_frontier)):
            node = current_frontier[i]
            neighbors = adj_list[node]

            for j in range(len(neighbors)):
                neighbor = neighbors[j]
                if visited[neighbor] == 0:                      # if neighbor not visited yet 
                    next_frontier_flags[neighbor] = 1           # neighbor is to be visited on next frontier
                    next_distances[neighbor] = dist[node] + 1   # neighbor is this distance from source through current node

        # Sequentially apply the updates to global arrays
        new_frontier = []
        for i in range(n):
            if next_frontier_flags[i] == 1:
                visited[i] = 1
                dist[i] = next_distances[i]
                new_frontier.append(i)

        current_frontier = new_frontier

    return dist


# -----------------------------------------------
# Main benchmarking section
# -----------------------------------------------
if __name__ == "__main__":
    runs = 50
    n_nodes = 1000
    large_graph_mult = 15       # G_2 will be 15 times larger node-wise

    gen_time = time.time()
    G_1_numba = generate_graph_numba(n_nodes, connectivity_pct=0.5)
    G_2_numba = generate_graph_numba(n_nodes * large_graph_mult, connectivity_pct=0.5)
    print(f"Numba gen graph time: {time.time() - gen_time} s")

    start_node = 0 

    # Compile once to exclude compilation time
    _, compile_time = bfs_parallel(G_1_numba, start_node, n_nodes)
    _, compile_time2 = bfs_sequential(G_1_numba, start_node, n_nodes)
    print(f"Compiled BFS functions in {round(compile_time + compile_time2, 4)} s")

    par1_total, par2_total, seq1_total, seq2_total = 0, 0, 0, 0
    for run_num in range(runs):
        print(f"run {run_num}")
        _, pt1 = bfs_parallel(G_1_numba, start_node, n_nodes)
        _, pt2 = bfs_parallel(G_2_numba, start_node, n_nodes * large_graph_mult)
        par1_total += pt1
        par2_total += pt2

        _, st1 = bfs_sequential(G_1_numba, start_node, n_nodes)
        _, st2 = bfs_sequential(G_2_numba, start_node, n_nodes * large_graph_mult)
        seq1_total += st1
        seq2_total += st2

    print(f"! SEQUENTIAL ! Average time: {round(((seq1_total + seq2_total) / runs) * 1000, 2)} ms")
    print(f"! SEQUENTIAL ! G_1 time: {round((seq1_total / runs) * 1000, 2)} ms")
    print(f"! SEQUENTIAL ! G_2 time: {round((seq2_total / runs) * 1000, 2)} ms")

    print(f"! PARALLEL   ! Average time: {round(((par1_total + par2_total) / runs) * 1000, 2)} ms")
    print(f"! PARALLEL   ! G_1 time: {round((par1_total / runs) * 1000, 2)} ms")
    print(f"! PARALLEL   ! G_2 time: {round((par2_total / runs) * 1000, 2)} ms")
