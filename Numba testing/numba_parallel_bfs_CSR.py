from numba import prange, njit
import numba
import numpy as np
import time
from collections import deque

from graph_generator import generate_graph
from bfs import bfs  # Reference sequential BFS for benchmarking

# -----------------------------------------------------------------------------------
# Decorator to time any function's execution and return the result + elapsed time
# -----------------------------------------------------------------------------------
def timer(f):
    def wrapper(*args, **kwargs):
        t_start = time.time()
        result = f(*args, **kwargs)
        t_end = time.time()
        return result, (t_end - t_start)
    return wrapper


# -----------------------------------------------------------------------------------
# Parallel BFS wrapper: handles conversion from Python graph (adjacency list)
# to a flat CSR-like format, which is much more suitable for Numba and SIMD access.
# -----------------------------------------------------------------------------------
def parallel_bfs(graph: dict[int, list[int]], start_node: int) -> list[tuple[int, int]]:
    """
    Converts a Python graph dict to a flat memory layout (CSR-like),
    and launches the BFS implemented in Numba.

    Parameters:
        graph (dict): Adjacency list of the graph.
        start_node (int): The source node to begin BFS from.

    Returns:
        list of (node, distance) tuples and time taken for the operation.
    """
    max_edges = sum(len(neighbors) for neighbors in graph.values())   # total amount of neighbors / edges

    # Map node ids to contiguous indices to work with Numba arrays
    nodes = np.array(sorted(graph.keys()), dtype=np.int32)
    id_map = {node: i for i, node in enumerate(nodes)}  # dict with keys = nodes and values = index (from enumerate)

    # CSR-like graph representation
    index = np.zeros(len(nodes) + 1, dtype=np.int32)  # Start/end indices for each node's edge list
    edges = np.zeros(max_edges, dtype=np.int32)       # Flattened list of all edges

    edge_pos = 0
    for i, node in enumerate(nodes):
        neighbors = graph[node]
        index[i] = edge_pos
        for neigh in neighbors:
            edges[edge_pos] = id_map[neigh]
            edge_pos += 1
    index[len(nodes)] = edge_pos  # Sentinel for last node

    # Call Numba-parallel BFS
    dist_array, time_taken = bfs_parallel_numba(index, edges, id_map[start_node], len(nodes))

    # Convert result back to original node IDs
    return [(nodes[i], dist_array[i]) for i in range(len(nodes)) if dist_array[i] != -1], time_taken


# -----------------------------------------------------------------------------------
# Numba-compiled BFS with parallel edge expansion
# -----------------------------------------------------------------------------------
@timer  # Measure time taken
@njit(parallel=True)  # Compile with Numba and enable parallel loops
def bfs_parallel_numba(index, edges, start_node_id, n):
    """
    Parallel BFS using level-synchronous strategy.
    Each BFS level is processed in parallel by looping over the current frontier.

    Parameters:
        index (np.ndarray): CSR start-end indices of each node’s adjacency list.
        edges (np.ndarray): Flattened adjacency lists of all nodes.
        start_node_id (int): The BFS starting point (in 0-based CSR space).
        n (int): Total number of nodes.

    Returns:
        dist (np.ndarray): Distance to each node from start, -1 if not reachable.
    """
    # Mark visited nodes (0 = unvisited, 1 = visited)
    visited = np.zeros(n, dtype=np.uint8)

    # Distance from start node, -1 means unvisited
    dist = -np.ones(n, dtype=np.int32)

    # Initialize BFS with the start node
    visited[start_node_id] = 1
    dist[start_node_id] = 0
    current_frontier = [start_node_id]

    # Continue until all reachable nodes are explored
    while len(current_frontier) > 0:
        # Temporary arrays to mark discovered nodes this round
        next_frontier_flags = np.zeros(n, dtype=np.uint8)
        next_distances = np.full(n, -1, dtype=np.int32)

        # -------------------- PARALLEL SECTION --------------------
        # For each node in the current frontier, discover its neighbors
        # Each iteration of `prange` runs in parallel on different threads
        for i in prange(len(current_frontier)):
            node = current_frontier[i]
            start_edge = index[node]
            end_edge = index[node + 1]

            for j in range(start_edge, end_edge):
                neighbor = edges[j]

                # Avoid race condition: do not mutate shared arrays here
                if visited[neighbor] == 0:
                    next_frontier_flags[neighbor] = 1  # Mark for next round
                    next_distances[neighbor] = dist[node] + 1

        # -------------------- SEQUENTIAL SECTION --------------------
        # Apply updates to global state in a single thread to avoid data races
        new_frontier = []
        for i in range(n):
            if next_frontier_flags[i] == 1 and visited[i] == 0:
                visited[i] = 1
                dist[i] = next_distances[i]
                new_frontier.append(i)

        current_frontier = new_frontier  # Prepare for next level

    return dist


# -----------------------------------------------------------------------------------
# Benchmarking setup — compare sequential vs parallel BFS
# -----------------------------------------------------------------------------------
if __name__ == "__main__":
    runs = 1000
    n_nodes = 2500
    n_edges = n_nodes * 5

    # Generate two graphs: one small, one much larger
    G_1 = generate_graph(n_nodes, n_edges)
    G_2 = generate_graph(15 * n_nodes, 15 * n_edges)

    start_node = 0  # BFS source node

    # Time accumulators
    seq_total_time = 0
    seq_first_timer = 0
    seq_second_timer = 0

    par_total_time = 0
    par_first_timer = 0
    par_second_timer = 0

    # run 1 time to NOT time compilation time
    _, compile_time = parallel_bfs(G_1, start_node)
    print(f"Compiled `parallel_bfs` in {compile_time} s")

    for _ in range(runs):
        # Parallel BFS timings
        result1, par_time1 = parallel_bfs(G_1, start_node)
        _, par_time2 = parallel_bfs(G_2, start_node)
        par_first_timer += par_time1
        par_second_timer += par_time2
        par_total_time += par_time1 + par_time2

        # print(result1)

        # Sequential BFS timings
        result2, seq_time1 = bfs(G_1, start_node)
        _, seq_time2 = bfs(G_2, start_node)
        seq_first_timer += seq_time1
        seq_second_timer += seq_time2
        seq_total_time += seq_time1 + seq_time2

        # print(result2)

    # -------------------- Output summary --------------------
    print(f"Executed both functions {runs} times:")

    print(f"! SEQUENTIAL !")
    print(f"\tTotal time: {round(seq_total_time, 5)} seconds")
    print(f"\taverage time: {round((seq_total_time / runs) * 1000, 2)} ms")
    print(f"\tTime for first graph: {round(seq_first_timer, 5)} seconds")
    print(f"\tTime for second graph: {round(seq_second_timer, 5)} seconds")

    print(f"! PARALLEL !")
    print(f"\tTotal time: {round(par_total_time, 5)} seconds")
    print(f"\taverage time: {round((par_total_time / runs) * 1000, 2)} ms")
    print(f"\tTime for first graph: {round(par_first_timer, 5)} seconds")
    print(f"\tTime for second graph: {round(par_second_timer, 5)} seconds")