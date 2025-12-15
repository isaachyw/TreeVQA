# Solve the ground state of the power grid graph using NumPyMinimumEigensolver from Qiskit and store results in a JSON file

import json
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit.quantum_info import SparsePauliOp
from ieee14 import build_ieee14_graph_family
from graphs_utils import build_max_cut_paulis
from numpy import arange
import matplotlib.pyplot as plt
import scienceplots


plt.style.use(["science"])


def compute_ground_states(
    load_scales, output_json="../../../ground-state/power-grid.json"
):
    """
    For each load_scale, build the IEEE14 graph, convert to MaxCut Pauli operator,
    solve for the ground state energy using NumPyMinimumEigensolver, and store results.
    """
    graphs = build_ieee14_graph_family(load_scales)
    energies = []
    results = {}
    for idx, graph in enumerate(graphs):
        pauli_list = build_max_cut_paulis(graph)
        op = SparsePauliOp.from_list(pauli_list)
        solver = NumPyMinimumEigensolver()
        result = solver.compute_minimum_eigenvalue(op)
        ground_energy = result.eigenvalue.real
        results[str(round(load_scales[idx], 3))] = {
            "ground_energy": ground_energy,
        }
        energies.append(ground_energy)
        print(f"Load scale {load_scales[idx]}: ground energy = {ground_energy:.6f}")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"Results written to {output_json}")
    # plot the energies
    plt.plot(load_scales, energies)
    plt.savefig("../../../ground-state/power-grid.png")


if __name__ == "__main__":
    # Example: compute for load scales 1.0, 1.5, 2.0, 2.5, 3.0
    load_scales = arange(0.5, 2, 0.01)
    compute_ground_states(load_scales)
