from mpi4py import MPI
import numpy as np
from time import time

# The graph generator from A.
from numba_graph import generate_graph_numba

# Run command
# mpiexec -n 2 python bfs_mpi_final.py

# ---------- Setup communication -----------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world_size = comm.Get_size()

# ------------------------------------------


# ---------- Initialize variables ----------

# Queue containing the tasks for rank == 0.
Q_tasks = []
start_node = 0
Q_tasks.append(start_node)

# Initialize variables
graph = None
distances = None
tasks = None
terminate = None
# ------------------------------------------


# ---------- Create and distribute the graph -----------
if rank == 0:
    # Set node amount
    nodeAmount = 300
    print(f"Node amount: {nodeAmount}")
    print(f"Set node amount to 4500 for graph G_2")

    # Create the distances array
    distances = -np.ones(nodeAmount, np.int32)
    distances[start_node] = 0

    # Create graph and convert from numba array to list.
    graph = list(generate_graph_numba(nodeAmount))
    for i in range(len(graph)):
        graph[i] = list(graph[i])
    #print(graph)
    t1 = time()

# Broadcast the graph from rank 0.
graph = comm.bcast(graph, root=0)
# -------------------------------------------------------


while True:
    # Reset the frontier
    Q_frontier = []

    # ---------- Partition, scatter and broadcast -----------
    if rank == 0:
        # Reset the tasks array.
        tasks = [[] for _ in range(world_size)]
        
        # Partition the nodes among all processes.
        for i in range(len(Q_tasks)):
            tasks[i % world_size].append(Q_tasks[i])
    # Scatter and broadcast queue and distances.
    Q_tasks = comm.scatter(tasks, root=0)
    distances = comm.bcast(distances, root=0)
    # -------------------------------------------------------


    # ------------- Perform BFS on each process -------------

    # Go through all nodes in the task
    for node in Q_tasks:
        # Go through all neighbors of that node.
        for neighbor in graph[node]:

            # Check if the neighbor has already been visited
            if distances[neighbor] != -1:
                continue
            
            # Add the neighbor node to the new frontier and 
            # add 1 to their distances to the source.
            Q_frontier.append(neighbor)
            distances[neighbor] = distances[node] + 1

    # Gather the results on rank 0
    Q_frontier = comm.gather([Q_frontier, distances], root=0)
    # -------------------------------------------------------


    # ----------- Combine frontiers and distances -----------
    if rank == 0:
        # Clear the task array
        Q_tasks.clear()

        # Combine distances and frontiers from all processes
        for i in range(len(Q_frontier)):

            # Combine the distance arrays
            for m in range(1, len(Q_frontier[i][1])):
                if Q_frontier[i][1][m] > 0:
                    distances[m] = Q_frontier[i][1][m]

            # Create the new tasks array.
            for m in range(len(Q_frontier[i][0])):
                if Q_frontier[i][0][m] != m:
                    if Q_frontier[i][0][m] not in Q_tasks:
                        Q_tasks.append(Q_frontier[i][0][m])
    # -------------------------------------------------------


    # ------------ Check termination condiation -------------
    # Check if size of the task queue on rank 0
    if rank == 0:
        terminate = len(Q_tasks)
    
    # Broadcast the size of the task queue
    terminate = comm.bcast(terminate, root=0)

    # The program is finished if the task queue is empty.
    if terminate == 0:
        break
    # -------------------------------------------------------

# Print results
if rank == 0:
    t2 = time()
    print(f"MPI average runtime: {round((t2 - t1) * 1000, 2)} ms")
    #print(f"Distances: {distances}")
