from igraph import Graph
import random

def generate_connected_graph_igraph(num_nodes: int, avg_degree: int) -> dict[int, list[int]]:
    """
    Generates a connected undirected graph using igraph with a given average degree.

    Parameters:
        num_nodes (int): Number of nodes in the graph.
        avg_degree (int): Average number of edges per node.

    Returns:
        dict: Adjacency list representation of the graph.
    """
    # Estimate number of edges for the given average degree
    num_edges = (num_nodes * avg_degree) // 2  # undirected, so divide by 2

    # Use the Erdős–Rényi model to generate a random graph
    while True:
        g: Graph = Graph.Erdos_Renyi(n=num_nodes, m=num_edges, directed=False, loops=False)
        if g.is_connected():
            break

    # Convert igraph format to a standard adjacency list (int -> list[int])
    adjacency_list = {
        v.index: [nbr.index for nbr in g.vs[v.index].neighbors()] for v in g.vs
    }
    return adjacency_list


if __name__ == "__main__":
    graph = generate_connected_graph_igraph(num_nodes=100, avg_degree=6)
    for k, v in list(graph.items())[:5]:  # Preview first few nodes
        print(f"{k}: {v}")
