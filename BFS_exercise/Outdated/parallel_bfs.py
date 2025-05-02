import multiprocessing as mp
from multiprocessing import Pool
import time
from concurrent.futures import ThreadPoolExecutor

from multiprocessing.shared_memory import ShareableList

from igraph_generator import generate_connected_graph_igraph as generate_graph  # Import the graph generation function

def timer(f):
    # Make sure the input arguments from the function f
    # is passed as well.
    def wrapper(*args, **kwargs):
        # Time the function as input
        t_start = time.time()
        result = f(*args, **kwargs)
        t_end = time.time()

        # Return the result AND the execution time
        return result, (t_end - t_start)
    return wrapper


def worker(args):

    # Pass worker arguments
    nodes = args[0]
    distances = args[1]
    graph = args[2]
    Q: mp.Queue = args[3]
    mutex: mp.mutex = args[4]

    # Go through each node in the task.
    for node in nodes:
            # Go through all neighbors to that node.
            for neighbor in graph[node]:
                # Make sure we have the mutex
                with mutex:
                    if distances[neighbor] != -1:
                        continue
                    Q.put(neighbor)
                    distances[neighbor] = distances[node] + 1
    

@timer
def bfs_parallel(graph, start_node):
    mp.freeze_support() # For Windows compatibility
    mp.set_start_method('spawn', force=True)  # Set the start method for multiprocessing
    processors = mp.cpu_count()
    processors = 5
    pool = mp.Pool(processors)  # Create a pool of worker processes

    # Create the Queue, distance list, and mutex.
    Q = mp.Manager().Queue()

    distances = ShareableList([-1 for _ in range(len(graph))], name="Distances")

    # Create a mutex
    mutex = mp.Manager().Lock()

    # Add the starting node to the queue
    Q.put(start_node)
    distances[start_node] = 0

    while not Q.empty():
        tasks = [[] for _ in range(processors)]
        
        # Get amount of nodes
        nodeAmount = Q.qsize()

        # Create distribution of nodes
        for i in range(nodeAmount):
            tasks[i % processors].append(Q.get())
        tasks = [(i, distances, graph, Q, mutex) for i in tasks]
        
        # Create the pool for mapping tasks to workers.
        pool.map(worker, tasks)

    #synchronize the workers
    pool.close()
    pool.join()


def main(runs: int):
    # Parameters for graph generation
    num_nodes = 10  # Number of nodes in the graph
    num_edges = round(num_nodes * 0.8)  # Number of edges in the graph
    num_nodes2 = num_nodes * 15  # Number of nodes in the second graph
    num_edges2 = num_edges * 15  # Number of edges in the second graph
    
    # Generate graph 1
    print("Generating graph...")
    graph = generate_graph(num_nodes, num_edges)
    print(f"Graph generated with {num_nodes} nodes and {num_edges} edges.")

    # Generate graph 2
    print("Generating second graph...")
    graph2 = generate_graph(num_nodes2, num_edges2)
    print(f"Second graph generated with {num_nodes2} nodes and {num_edges2} edges.")

    #print(graph)
    #print(graph2)

    # Perform BFS and measure elapsed time
    start_node = 0  # Starting node for BFS
    
    total_time = 0  # time spent running bfs()
    first_timer = 0  # time spent running bfs() on the first graph
    second_timer = 0  # time spent running bfs() on the second graph
    for _ in range(runs):
        
        distance1, time_taken = bfs_parallel(graph, start_node)
        distance2, time_taken2 = bfs_parallel(graph2, start_node)
        first_timer += time_taken
        second_timer += time_taken2
        total_time += time_taken + time_taken2
        

        #print(f"Distances for first graph: {distance1}")
        #print(f"Distances for second graph: {distance2}")
        # # Print the graph structure
        # print("Any nodes left out?:")
        # for node, neighbors in graph.items():
        #     if len(neighbors) == 0:
        #         print(f"Node {node}: {neighbors}")
        # # Output results
        # print(f"BFS completed. Number of nodes visited: {len(bfs_result)}")
    
    print(f"Executed bfs() {runs} times:")
    print(f"\tTotal time: {round(total_time, 5)} seconds")
    print(f"\taverage time: {round((total_time / runs) * 1000, 2)} ms")
    print(f"\tTime for first graph: {round(first_timer, 5)} seconds")
    print(f"\tTime for second graph: {round(second_timer, 5)} seconds")

if __name__ == "__main__":
    runs = 1  # Number of runs for the BFS function
    main(runs=runs)