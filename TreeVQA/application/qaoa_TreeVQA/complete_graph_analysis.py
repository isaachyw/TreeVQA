import rustworkx as rx
import numpy as np
from qiskit.quantum_info import SparsePauliOp
import matplotlib.pyplot as plt
import json
import os
from typing import List, Dict, Tuple
from .graphs_utils import build_max_cut_paulis
from scipy.sparse.linalg import eigsh
# import cupy as cp
# from cupyx.scipy.sparse.linalg import eigsh as cupy_eigsh


def generate_complete_graph_family(
    num_vertices: int = 10, num_graphs: int = 100
) -> List[rx.PyGraph]:
    """Generate a family of complete graphs with systematic edge weights."""
    graphs = []
    # Define weight range and step size
    weight_min = 0.5
    weight_max = 1.5
    step_size = 0.05  # Smaller step size for finer precision

    for i in range(num_graphs):
        G = rx.PyGraph()
        G.add_nodes_from(range(num_vertices))

        # Systematic weight assignment based on graph index
        for j in range(num_vertices):
            for k in range(j + 1, num_vertices):
                # Use a deterministic weight based on edge position and graph index
                # Map the weight to the desired range with finer precision
                base_weight = (
                    (i + 1) * (j + k + 1) % 100
                )  # Use modulo 100 for more variation
                normalized_weight = base_weight / 100  # Normalize to [0,1]
                weight = weight_min + normalized_weight * (weight_max - weight_min)
                # Round to nearest step size
                weight = round(weight / step_size) * step_size
                G.add_edge(j, k, weight)
        graphs.append(G)
    return graphs


# def solve_ground_state(graph: rx.PyGraph) -> Tuple[float, np.ndarray]:
#     """Solve the ground state energy of a graph's MaxCut Hamiltonian using GPU."""
#     # Convert graph to Pauli operators
#     pauli_list = build_max_cut_paulis(graph)

#     # Create SparsePauliOp
#     paulis = [p[0] for p in pauli_list]
#     coeffs = [p[1] for p in pauli_list]
#     sp_op = SparsePauliOp(data=paulis, coeffs=coeffs)

#     # Convert to sparse matrix and move to GPU
#     matrix = sp_op.to_matrix(sparse=True)
#     matrix_gpu = cp.sparse.csr_matrix(matrix)

#     # Solve on GPU
#     eigenvalues, eigenvectors = cupy_eigsh(matrix_gpu, k=1, which="SA")

#     # Move results back to CPU
#     eigenvalue = eigenvalues[0].get().real
#     eigenvector = eigenvectors[:, 0].get()

#     return eigenvalue, eigenvector


def analyze_graph_family():
    """Analyze a family of complete graphs and save results."""
    # Create output directory if it doesn't exist
    os.makedirs("ground-state", exist_ok=True)

    # Generate graphs
    num_vertices = 10
    num_graphs = 100
    graphs = generate_complete_graph_family(num_vertices, num_graphs)

    # Analyze each graph
    results = []
    for i, graph in enumerate(graphs):
        print(f"Processing graph {i + 1}/{num_graphs}")
        energy, state = solve_ground_state(graph)

        # Calculate graph properties
        edge_weights = [graph.get_edge_data(e[0], e[1]) for e in graph.edge_list()]
        avg_weight = np.mean(edge_weights)
        max_weight = np.max(edge_weights)

        results.append(
            {
                "graph_index": i,
                "ground_energy": float(energy),
                "average_edge_weight": float(avg_weight),
                "max_edge_weight": float(max_weight),
                "edge_weights": edge_weights,
            }
        )

    # Save results
    with open("../ground-state/complete_graph_analysis.json", "w") as f:
        json.dump(results, f, indent=4)

    # Create visualizations
    plot_results(results)


def plot_results(results: List[Dict]):
    """Create visualizations of the analysis results."""
    # Extract data for plotting
    energies = [r["ground_energy"] for r in results]
    avg_weights = [r["average_edge_weight"] for r in results]
    max_weights = [r["max_edge_weight"] for r in results]

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot 1: Ground energy vs average edge weight
    ax1.scatter(avg_weights, energies, alpha=0.5)
    ax1.set_xlabel("Average Edge Weight")
    ax1.set_ylabel("Ground State Energy")
    ax1.set_title("Ground Energy vs Average Edge Weight")

    # Plot 2: Ground energy vs max edge weight
    ax2.scatter(max_weights, energies, alpha=0.5)
    ax2.set_xlabel("Maximum Edge Weight")
    ax2.set_ylabel("Ground State Energy")
    ax2.set_title("Ground Energy vs Maximum Edge Weight")

    plt.tight_layout()
    plt.savefig(
        "../ground-state/complete_graph_analysis.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


if __name__ == "__main__":
    analyze_graph_family()
