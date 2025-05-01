import random


def generate_graph(num_nodes: int, num_edges: int) -> dict[int, list]:
    """
    Generates a random graph represented as an adjacency list.

    Parameters:
        num_nodes (int): Number of nodes in the graph.
        num_edges (int): Number of edges in the graph.

    Returns:
        dict: A dictionary representing the adjacency list of the graph.
    """
    graph: dict[int, list] = {i: [] for i in range(num_nodes)}

    # Seed the random function for the sake of reproducibility.
    random.seed(42)

    # Generate random edges
    for _ in range(num_edges):

        # Generate two node connections.
        u = random.randint(0, num_nodes - 1)
        v = random.randint(0, num_nodes - 1)

        # Check the the nodes does not have the same index, 
        # and that they does not already have a connection.
        if (u != v) and (v not in graph[u]) and (u not in graph[v]):

            # Create the connection.
            graph[u].append(v)
            graph[v].append(u)

    return graph


if __name__ == '__main__':
    print(generate_graph(10, 10))

