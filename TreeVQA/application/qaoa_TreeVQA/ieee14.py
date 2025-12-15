# grid_zoning_maxcut_driver.py
# Realistic small-scale Max-Cut instance from the IEEE 14-bus grid (via pandapower).
# Weight = base-case line flow (MW or MVA), optionally after scaling loads.
# Optionally reduce to ~12 nodes by keeping top-N buses by weighted degree.
#
from typing import Iterable, List

import math
import numpy as np
import networkx as nx
import rustworkx as rx

import pandapower as pp
import pandapower.networks as pn
import matplotlib.pyplot as plt

# ---------- Build weighted graph from power flow --------------------------------


def build_ieee14_graph(
    load_scale=1.0, weight_metric="flow_mw", edge_threshold=0.0, restrict_to_n=None
):
    """
    Returns (labels, W, G, net) where:
      - labels: list of bus names in graph order
      - W: symmetric numpy array of edge weights for Max-Cut
      - G: networkx Graph with 'weight' on edges
      - net: pandapower network after running power flow

    Parameters
    ----------
    load_scale : float
        Multiply all P/Q loads by this factor to create a different operating point.
        (This is your *tunable parameter* → different flows → different weights.)
    weight_metric : {"flow_mw","flow_mva"}
        Use |P_line| (MW) or |S_line| (MVA) as edge weights.
    edge_threshold : float
        Drop edges with weight < threshold to sparsify the graph (optional).
    restrict_to_n : int or None
        If set (e.g., 12), keep top-N buses by weighted degree to hit a target size.
    """
    # 1) Load dataset
    net = pn.case14()  # IEEE 14-bus test case
    # 2) Scale loads to change stress and flows (the "knob")
    if load_scale != 1.0 and len(net.load):
        net.load["p_mw"] *= load_scale
        net.load["q_mvar"] *= load_scale

    # 3) Solve AC power flow
    pp.runpp(net, calculate_voltage_angles=True, numba=False)

    # 4) Build weighted bus graph from line results
    G = nx.Graph()
    # bus labels
    bus_labels = {int(b): f"bus_{int(b)}" for b in net.bus.index.values}
    for b in net.bus.index.values:
        G.add_node(str(b))

    # results per line
    res = net.res_line
    line = net.line

    for idx in line.index.values:
        fb = int(line.at[idx, "from_bus"])
        tb = int(line.at[idx, "to_bus"])
        name_u, name_v = str(fb), str(tb)

        p_from = abs(res.at[idx, "p_from_mw"])
        q_from = abs(res.at[idx, "q_from_mvar"])
        p_to = abs(res.at[idx, "p_to_mw"])
        q_to = abs(res.at[idx, "q_to_mvar"])

        if weight_metric == "flow_mva":
            s_from = math.hypot(p_from, q_from)
            s_to = math.hypot(p_to, q_to)
            w = 0.5 * (s_from + s_to)
        else:
            w = 0.5 * (p_from + p_to)  # default: MW

        if w >= edge_threshold and w > 0:
            G.add_edge(name_u, name_v, weight=float(w))

    # 5) Optionally restrict to N nodes (e.g., 12) by weighted degree
    if restrict_to_n is not None and restrict_to_n < G.number_of_nodes():
        wd = {
            n: sum(d.get("weight", 0.0) for _, _, d in G.edges(n, data=True))
            for n in G.nodes()
        }
        keep = sorted(wd, key=wd.get, reverse=True)[:restrict_to_n]
        G = G.subgraph(keep).copy()

    # 6) Build symmetric weight matrix W in graph order
    mapping = {"12": "6", "13": "7"}
    G = nx.relabel_nodes(G, mapping)
    return G


def build_ieee14_graph_family(load_scales: Iterable[float]) -> List[rx.PyGraph]:
    """
    Build a family of IEEE 14-bus graphs with different load scales.
    """
    graphs = []
    for load_scale in load_scales:
        G = build_ieee14_graph(load_scale=load_scale, restrict_to_n=12)
        G = rx.networkx_converter(G)
        graphs.append(G)
    return graphs


def to_unweighted(graph, unit_weight=1):
    # Pick the right graph type
    if isinstance(graph, rx.PyDiGraph):
        H = rx.PyDiGraph(multigraph=graph.multigraph)
    elif isinstance(graph, rx.PyGraph):
        H = rx.PyGraph(multigraph=graph.multigraph)
    else:
        raise TypeError("Expected a PyGraph or PyDiGraph")

    # Copy nodes in index order so indices match
    for data in graph.nodes():
        H.add_node(data)

    # Add the same edges but with unit weights
    for u, v in graph.edge_list():
        H.add_edge(u, v, unit_weight)

    return H


def build_ieee14_unweighted_graph():
    G = build_ieee14_graph(load_scale=1.0, restrict_to_n=12)
    G = rx.networkx_converter(G)
    G = to_unweighted(G)
    return G


def total_variance(graphs, weight_fn=None):
    """
    Compute the total variance across a list of isomorphic rustworkx graphs.

    Parameters
    ----------
    graphs : list of rustworkx.PyGraph or rustworkx.PyDiGraph
        Input graphs (assumed to be isomorphic and with the same number of nodes).
    weight_fn : callable, optional
        A function edge -> weight. If None, assumes edge weight is stored directly.

    Returns
    -------
    total_var : float
        The total variance of edge weights across graphs.
    """
    if not graphs:
        raise ValueError("Empty list of graphs")

    n = graphs[0].num_nodes()
    K = len(graphs)

    # Collect adjacency matrices
    mats = []
    for g in graphs:
        A = np.zeros((n, n), dtype=float)
        for u, v, w in g.weighted_edge_list():
            breakpoint()
            A[u, v] = w if weight_fn is None else weight_fn(w)
            if not g.is_directed():
                A[v, u] = A[u, v]
        mats.append(A)

    mats = np.stack(mats, axis=0)  # (K, n, n)

    # Mean adjacency matrix
    mean_mat = mats.mean(axis=0)

    # Frobenius variance
    total_var = np.sum((mats - mean_mat) ** 2) / (K - 1)
    return total_var


# ---------- Solve weighted Max-Cut with QAOA (Qiskit) ---------------------------


def qaoa_maxcut(W, reps=2, optimizer_maxiter=200, seed=123):
    from qiskit_optimization.applications import Maxcut
    from qiskit_optimization.algorithms import MinimumEigenOptimizer
    from qiskit_algorithms import QAOA
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit.primitives import Sampler

    qp = Maxcut(W).to_quadratic_program()
    qaoa = QAOA(
        sampler=Sampler(),
        optimizer=COBYLA(maxiter=optimizer_maxiter),
        reps=reps,
    )
    result = MinimumEigenOptimizer(qaoa).solve(qp)
    return result  # result.x (0/1), result.fval (cut), result.status


# ---------- Example CLI ---------------------------------------------------------
if __name__ == "__main__":
    G1 = build_ieee14_graph_family(np.arange(0.5, 1.5, 0.1))
    G2 = build_ieee14_graph_family(np.arange(0.8, 1.2, 0.04))
    G3 = build_ieee14_graph_family(np.arange(0.9, 1.1, 0.02))
    var1 = total_variance(G1)
    var2 = total_variance(G2)
    var3 = total_variance(G3)
    print(var1, var2, var3)

if __name__ == "__main__2":
    # Make ~12-node instance by keeping the 12 most “active” buses
    G = build_ieee14_graph(
        load_scale=2.5,  # <-- tunable knob: stress system to reshape weights
        weight_metric="flow_mw",  # or "flow_mva"
        # edge_threshold=0.0,
        restrict_to_n=12,  # target problem size
    )
    print(f"Graph: |V|={len(G.nodes())}, |E|={G.number_of_edges()}")
    # print the node labels
    # 4. Visualize the resulting graph using NetworkX and Matplotlib
    # plt.figure(figsize=(10, 8))
    # Make nodes smaller and edges longer by adjusting node_size and spring_layout parameters
    pos = nx.spring_layout(G, k=10, seed=42)  # increase k for longer edges
    edge_labels = nx.get_edge_attributes(G, "weight")
    # round each edge label to 2 decimal places
    edge_labels = {k: round(v, 2) for k, v in edge_labels.items()}
    print(edge_labels)
    print(G.edges())
    plt.figure(figsize=(12, 10))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="lightblue",
        node_size=1000,  # smaller nodes
        # font_size=10,
        # width=1.5,
    )
    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        # font_size=8,
    )
    plt.savefig("graph.png")
    try:
        res = qaoa_maxcut(W, reps=2, optimizer_maxiter=200)
        print("QAOA Max-Cut value:", res.fval)
        x = np.array(res.x)
        S = [labels[i] for i in np.where(x == 1)[0]]
        Sc = [labels[i] for i in np.where(x == 0)[0]]
        print(f"|S|={len(S)}  |S^c|={len(Sc)}")
        print("S :", S)
    except Exception as e:
        print("QAOA step skipped (install qiskit* packages).", e)
