from mpi4py import MPI
from BFS_exercise.Outdated.igraph_generator import generate_connected_graph_igraph as igenerate_graph  # Import the igraph generation function
from graph_generator import generate_graph  # Import the graph generation function
import numpy as np
import sys

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

    # Fixed graph for fair benchmarking
    if rank == 0:
        graph = generate_graph(5000, 25000)
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
