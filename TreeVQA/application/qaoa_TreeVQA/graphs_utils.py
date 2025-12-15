import rustworkx as rx
import random
import networkx as nx


def generate_k_regular_graph(num_vertices, k, weighted=False, seed=False):
    """
    Generate a k-regular graph with a specified number of vertices.

    Parameters
    ----------
    num_vertices : int
        The number of vertices in the graph.
    k : int
        Each vertex will have exactly k edges. Must be less than num_vertices and num_vertices * k must be even.
    weighted : bool, optional
        If True, the edges will have random weights between 1 and 10. If False, all edges will have a weight of 1. Default is False.
    seed : bool, optional
        If True, the random number generator will be seeded with 0 for reproducibility. Default is False.

    Returns
    -------
    graph : rx.PyGraph
        A k-regular graph with the specified number of vertices and edges.

    Raises
    ------
    ValueError
        If k is greater than or equal to num_vertices or if num_vertices * k is not even.

    Notes
    -----
    The function uses a retry mechanism to ensure that a valid k-regular graph is generated. If a deadlock is detected during edge assignment, the process restarts from scratch.
    """
    if seed:
        random.seed(0)

    if k >= num_vertices or (num_vertices * k) % 2 != 0:
        raise ValueError("Invalid parameters: k must be < n and n*k must be even")

    while True:  # Retry if we fail to complete the graph
        graph = rx.PyGraph()
        graph.add_nodes_from(range(num_vertices))

        stubs = list(range(num_vertices)) * k
        random.shuffle(stubs)

        edges = set()
        while stubs:
            u, v = stubs.pop(), stubs.pop()
            attempts = 0
            while u == v or (u, v) in edges or (v, u) in edges:
                stubs.append(v)
                random.shuffle(stubs)
                v = stubs.pop()
                attempts += 1
                if attempts > len(stubs):  # Deadlock detected, restart
                    break
            else:
                edges.add((u, v))
                weight = random.randint(1, 10) if weighted else 1
                graph.add_edge(u, v, weight)
                continue
            break  # Restart from scratch

        if len(edges) == (num_vertices * k) // 2:
            return graph  # Successfully created a k-regular graph


def generate_random_complete_graph(
    num_vertices, weighted=False, seed=False, save_path=None
):
    """
    Generate a random complete graph.
    Parameters
    ----------
    num_vertices : int
        The number of vertices in the graph.
    weighted : bool, optional
        If True, edges will have random weights between 1 and 10. Default is False.
    seed : bool, optional
        If True, the random seed will be set to 0 for reproducibility. Default is False.
    save_path : str, optional
        Path to save the generated graph. Default is None.
    Returns
    -------
    G : rx.PyGraph
        The generated complete graph.
    """
    G = rx.PyGraph()
    G.add_nodes_from(range(num_vertices))

    if seed:
        random.seed(0)

    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            weight = random.randint(1, 10) if weighted else 1
            G.add_edge(i, j, weight)
    return G


def build_max_cut_paulis(graph: rx.PyGraph) -> list[tuple[str, float]]:
    """Convert the graph to Pauli list.

    This function does the inverse of `build_max_cut_graph`
    """
    pauli_list = []
    for edge in list(graph.edge_list()):
        paulis = ["I"] * len(graph)
        paulis[edge[0]], paulis[edge[1]] = "Z", "Z"

        weight = graph.get_edge_data(edge[0], edge[1])
        if isinstance(weight, dict):
            weight = weight.get("weight", 1)

        pauli_list.append(("".join(paulis)[::-1], weight))

    return pauli_list


def rustworkx_to_networkx(rw_graph):
    """
    Convert a rustworkx graph (PyGraph or PyDiGraph) into a networkx graph.

    Parameters:
        rw_graph (rustworkx.PyGraph or rustworkx.PyDiGraph): The rustworkx graph.

    Returns:
        networkx.Graph or networkx.DiGraph: The equivalent networkx graph.
    """
    if isinstance(rw_graph, rx.PyDiGraph):
        nx_graph = nx.DiGraph()
    elif isinstance(rw_graph, rx.PyGraph):
        nx_graph = nx.Graph()
    else:
        raise TypeError("Input must be a rustworkx.PyGraph or rustworkx.PyDiGraph")

    # Add nodes with data
    for index, data in enumerate(rw_graph.nodes()):
        nx_graph.add_node(index, data=data)

    # Add edges with data and weight
    for edge in rw_graph.weighted_edge_list():
        u, v, w = edge
        edge_data = rw_graph.get_edge_data(u, v)
        nx_graph.add_edge(u, v, data=edge_data, weight=w)

    return nx_graph
