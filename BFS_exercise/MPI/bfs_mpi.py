from mpi4py import MPI
from graph_generator import generate_graph  # Import the graph generation function
import numpy as np
import sys
from numba import njit, prange
from numba.typed import List as NumbaList

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

def partition_graph(graph, size):
    node_to_rank = {}
    for node in graph:
        node_to_rank[node] = node % size
    return node_to_rank

def bfs_mpi(graph, start_node):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    node_to_rank = partition_graph(graph, size)
    local_graph = {node: neighbors for node, neighbors in graph.items() if node_to_rank[node] == rank}
    visited = set()
    local_frontier = [start_node] if node_to_rank[start_node] == rank else []

    distance = {}
    if start_node in local_frontier:
        visited.add(start_node)
        distance[start_node] = 0

    level = 0
    while True:
        send_buffers = [[] for _ in range(size)]

        for node in local_frontier:
            for neighbor in graph[node]:
                owner = node_to_rank[neighbor]
                if neighbor not in visited:
                    send_buffers[owner].append((neighbor, level + 1))
                    if owner == rank:
                        visited.add(neighbor)
                        distance[neighbor] = level + 1

        recv_buffers = MPI.COMM_WORLD.alltoall(send_buffers)
        new_frontier = []
        for recv in recv_buffers:
            for neighbor, dist in recv:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distance[neighbor] = dist
                    new_frontier.append(neighbor)

        total_new = MPI.COMM_WORLD.allreduce(len(new_frontier), op=MPI.SUM)
        if total_new == 0:
            break

        local_frontier = new_frontier
        level += 1

    return distance

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    n_nodes = 300

    # Fixed graph for fair benchmarking
    if rank == 0:
        numba_graph = generate_graph_numba(n_nodes, connectivity_pct=0.5)
        graph = {}
        for node_id, adjlist in enumerate(numba_graph):
            graph[node_id] = list(adjlist[:])
    else:
        graph = None

    graph = comm.bcast(graph, root=0)

    # Benchmark BFS
    start_node = 0
    comm.Barrier()
    t_start = MPI.Wtime()
    bfs_mpi(graph, start_node)
    t_end = MPI.Wtime()

    total_time = t_end - t_start

    if rank == 0:
        print(f"[{size} processes] Time: {total_time:.4f} seconds")

if __name__ == "__main__":
    main()
