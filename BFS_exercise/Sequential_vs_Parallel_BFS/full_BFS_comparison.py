from numba import njit, prange
import numpy as np
import time
from numba.typed import List as NumbaList

from bfs import bfs                                 # Sequential BFS for reference
from parallel_bfs import bfs_parallel as mp_bfs     # MultiProcessing bfs for reference
from Numba_BFS_comparison import bfs_sequential as numba_sequential_bfs

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
def generate_graph_numba(num_nodes: int, connectivity_pct: float = 0.5):
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
def bfs_parallel_numba_adjlist(adj_list, start_node_id, n):
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
            if next_frontier_flags[i] == 1 and visited[i] == 0:
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
    n_nodes = 300
    large_graph_mult = 15       # G_2 will be 15 times larger node-wise

    gen_time = time.time()
    G_1_numba = generate_graph_numba(n_nodes, connectivity_pct=0.5)
    G_2_numba = generate_graph_numba(n_nodes * large_graph_mult, connectivity_pct=0.5)
    print(f"Numba gen graph time: {time.time() - gen_time} s")


    gen_time = time.time()
    # convert numba graphs to dict adjecency lists
    G_1_dict = {}
    for node_id, adjlist in enumerate(G_1_numba):
        G_1_dict[node_id] = list(adjlist[:])
    G_2_dict = {}
    for node_id, adjlist in enumerate(G_2_numba):
        G_2_dict[node_id] = list(adjlist[:])
    print(f"Numba graph to python dict conversion time: {time.time() - gen_time} s")

    start_node = 0

    # Compile once to exclude compilation time
    _, compile_time = bfs_parallel_numba_adjlist(G_1_numba, start_node, n_nodes)
    _, compile_time2 = numba_sequential_bfs(G_1_numba, start_node, n_nodes)
    print(f"Compiled BFS in {round(compile_time + compile_time2, 4)}s")

    par1_total, par2_total, seq1_total, seq2_total, mp1_total, mp2_total, seq3_total, seq4_total = 0, 0, 0, 0, 0, 0, 0, 0
    for run_num in range(runs):
        print(f"run {run_num}")
        _, pt1 = bfs_parallel_numba_adjlist(G_1_numba, start_node, n_nodes)
        _, pt2 = bfs_parallel_numba_adjlist(G_2_numba, start_node, n_nodes * large_graph_mult)
        par1_total += pt1
        par2_total += pt2

        _, st1 = bfs(G_1_dict, start_node)
        _, st2 = bfs(G_2_dict, start_node)
        seq1_total += st1
        seq2_total += st2

        _, st3 = numba_sequential_bfs(G_1_numba, start_node, n_nodes)
        _, st4 = numba_sequential_bfs(G_2_numba, start_node, n_nodes * large_graph_mult)
        seq3_total += st3
        seq4_total += st4

        _, mpt1 = mp_bfs(G_1_dict, start_node)
        _, mpt2 = mp_bfs(G_2_dict, start_node)
        mp1_total += mpt1
        mp2_total += mpt2

    print(f"! SEQUENTIAL PYTHON ! Average time: {round(((seq1_total + seq2_total) / runs) * 1000, 2)} ms")
    print(f"! SEQUENTIAL PYTHON ! G_1 time: {round((seq1_total / runs) * 1000, 2)} ms")
    print(f"! SEQUENTIAL PYTHON ! G_2 time: {round((seq2_total / runs) * 1000, 2)} ms")

    print(f"! SEQUENTIAL NUMBA  ! Average time: {round(((seq3_total + seq4_total) / runs) * 1000, 2)} ms")
    print(f"! SEQUENTIAL NUMBA  ! G_1 time: {round((seq3_total / runs) * 1000, 2)} ms")
    print(f"! SEQUENTIAL NUMBA  ! G_2 time: {round((seq4_total / runs) * 1000, 2)} ms")

    print(f"! PARALLEL NUMBA    ! Average time: {round(((par1_total + par2_total) / runs) * 1000, 2)} ms")
    print(f"! PARALLEL NUMBA    ! G_1 time: {round((par1_total / runs) * 1000, 2)} ms")
    print(f"! PARALLEL NUMBA    ! G_2 time: {round((par2_total / runs) * 1000, 2)} ms")

    print(f"! MULTIPROCESSING   ! Average time: {round(((mp1_total + mp2_total) / runs) * 1000, 2)} ms")
    print(f"! MULTIPROCESSING   ! G_1 time: {round((mp1_total / runs) * 1000, 2)} ms")
    print(f"! MULTIPROCESSING   ! G_2 time: {round((mp2_total / runs) * 1000, 2)} ms")
