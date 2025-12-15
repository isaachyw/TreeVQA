from typing import List
from qiskit import QuantumCircuit
import numpy as np


# Function to build the ma-QAOA circuit
def multi_angle_qaoa_circuit(gamma_params, beta_params, num_qubits, G, reps):
    qc = QuantumCircuit(num_qubits)
    qc.h(range(num_qubits))

    for rep in range(reps):
        gamma_offset = rep * len(G.edge_list())
        beta_offset = rep * len(G.node_indexes())

        for idx, (i, j, weight) in enumerate(G.weighted_edge_list()):
            if isinstance(weight, dict):
                weight = weight.get("weight", 1)
            gamma = gamma_params[gamma_offset + idx]  # Get the gamma parameter
            qc.cx(i, j)
            qc.rz(-2 * weight * gamma, j)
            qc.cx(i, j)

        for idx, i in enumerate(G.node_indexes()):
            beta = beta_params[beta_offset + idx]  # Get the beta parameter
            qc.rx(2 * beta, i)

    return qc


def assign_para_to_ma_ansatz(ma_ansatz, para, reps=2) -> List[float]:
    # x is [ 0.98467873,  1.25621434, -0.00867152,  0.29949082]
    beta_0 = 0.98467873
    gamma_0 = 1.25621434
    beta_1 = -0.00867152
    gamma_1 = 0.29949082
    x0 = []
    for para in ma_ansatz.parameters:
        para_name = para.name
        if para_name.startswith("beta"):
            if para_name.endswith("0"):
                x0.append(beta_0)
            elif para_name.endswith("1"):
                x0.append(beta_1)
        elif para_name.startswith("gamma"):
            if para_name.endswith("0"):
                x0.append(gamma_0)
            elif para_name.endswith("1"):
                x0.append(gamma_1)
    return x0
