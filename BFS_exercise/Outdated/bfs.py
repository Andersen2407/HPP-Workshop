import time
from collections import deque

from igraph_generator import generate_connected_graph_igraph as generate_graph

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


@timer
def bfs(graph: dict[int, list[int]], start_node: int) -> list[tuple[int, int]]:
    """
    Performs a Breadth-First Search (BFS) on the graph.

    Parameters:
        graph (dict): The adjacency list of the graph.
        start_node (int): The starting node for BFS.

    Returns:
        list: A list of nodes visited in BFS order and its distance to the source node.
    """
    visited: list[int] = [0 for _ in range(len(graph))]
    queue = deque([(start_node, 0)])
    bfs_order: list[tuple[int, int]] = []
    

    # Loop until the vertex queue is empty.
    while queue:
        
        # Mark the current node as visisted.
        entry = queue.popleft()
        node = entry[0]
        distance = entry[1]

        # Check if the node has already been visisted
        if visited[node] == 1: # O(1) time complexity
            continue

        # Add the node to the visited set.
        visited[node] = 1 # O(1) time complexity
        
        # Append the node and its distance to source.
        bfs_order.append((node, distance))

        # Do breadth first search to discover the neighbors of the node.
        for neighbor in graph[node]:

            # Check if the neighbors has already been visited
            if visited[neighbor] == 1:
                continue

            # Add the nodes neighbors that have not yet been visited to the queue.
            queue.append((neighbor, distance + 1))

    return bfs_order


def main(runs: int):
    # Parameters for graph generation
    num_nodes = 1000  # Number of nodes in the graph
    num_edges = round(num_nodes * 0.8)  # Number of edges in the graph
    num_nodes2 = num_nodes * 15  # Number of nodes in the second graph
    num_edges2 = num_edges * 15  # Number of edges in the second graph
    

    # Generate the graph
    print("Generating graph...")
    graph = generate_graph(num_nodes, num_edges)
    print(f"Graph generated with {num_nodes} nodes and {num_edges} edges.")
    #print(f'!!!!!!\nGraph:\n{graph}')
    print("Generating second graph...")
    graph2 = generate_graph(num_nodes2, num_edges2)
    print(f"Second graph generated with {num_nodes2} nodes and {num_edges2} edges.")

    #print(graph)

    # Perform BFS and measure elapsed time
    start_node = 0  # Starting node for BFS
    
    total_time = 0  # time spent running bfs()
    first_timer = 0  # time spent running bfs() on the first graph
    second_timer = 0  # time spent running bfs() on the second graph
    for _ in range(runs):
        
        bfs_result, time_taken = bfs(graph, start_node)
        bfs_result2, time_taken2 = bfs(graph2, start_node)
        first_timer += time_taken
        second_timer += time_taken2
        total_time += time_taken + time_taken2
        

        #print(f"BFS result for first graph: {bfs_result}")
        # print(f"BFS result for second graph: {bfs_result2}")
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
    runs = 1000  # Number of runs for the BFS function
    main(runs=runs)
    