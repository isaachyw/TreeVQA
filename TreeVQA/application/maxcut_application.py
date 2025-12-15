# Solve MaxCut with TreeVQA
"""MaxCut application for TreeVQA experiments."""

from typing import Dict, List
import numpy as np
import logging
import json
import os

from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import EfficientSU2
from qiskit.circuit import Parameter

from .treevqa_application import TreeVQAApplication
from ..op_task import OpTask
from .util import parse_slice, concatenate_slices
from .qaoa_TreeVQA.graphs_utils import build_max_cut_paulis
from .qaoa_TreeVQA.circuit_utils import multi_angle_qaoa_circuit
from .qaoa_TreeVQA.complete_graph_analysis import (
    generate_complete_graph_family,
)


class MaxCutExperiment(TreeVQAApplication):
    """TreeVQA experiment implementation for MaxCut problems"""

    DEFAULT_PARAMS = {
        **TreeVQAApplication.DEFAULT_PARAMS,
        "num_vertices": 10,
        "graph_slice": "0:100:1",
        "weight_range": "0.5:1.5:0.05",
    }

    def __init__(self, params: Dict):
        super().__init__(params)
        self.selected_graph_indices = []
        self.fig_params = {
            "x_label": "Graph Index",
            "title": f"MaxCut ({self.params['num_vertices']} vertices) Energy Landscape",
        }

    @property
    def result_dir_base(self) -> str:
        return f"{self.params['result_dir']}/MaxCut_{self.params['num_vertices']}vertices/{self.params['graph_slice']}"

    def load_data(self) -> None:
        selected_slices = parse_slice(self.params["graph_slice"], precision=3)
        self.selected_graph_indices = concatenate_slices(selected_slices)
        logging.info(
            "Selected %d graph indices: %s",
            len(self.selected_graph_indices),
            self.selected_graph_indices,
        )

        json_file = "ground-state/complete_graph_analysis.json"
        with open(json_file, "r", encoding="utf-8") as f:
            energy_map = json.load(f)
            try:
                self.ref_values = [
                    energy_map[int(idx)]["ground_energy"]
                    for idx in self.selected_graph_indices
                ]
            except (KeyError, IndexError):
                raise ValueError("Reference values not found in JSON file.")
        logging.info("Reference values: %s", [round(val, 5) for val in self.ref_values])

        graphs = generate_complete_graph_family(
            num_vertices=self.params["num_vertices"],
            num_graphs=int(max(self.selected_graph_indices)) + 1,
        )

        selected_graphs = [graphs[int(idx)] for idx in self.selected_graph_indices]

        pauli_ops = []
        for graph in selected_graphs:
            pauli_list = build_max_cut_paulis(graph)
            paulis = [p[0] for p in pauli_list]
            coeffs = [p[1] for p in pauli_list]
            pauli_ops.append(SparsePauliOp(data=paulis, coeffs=coeffs).sort())

        self.op_tasks = [
            OpTask(op.paulis, coeffs=op.coeffs, op_id=idx)
            for idx, op in zip(self.selected_graph_indices, pauli_ops)
        ]

        num_qubits = self.op_tasks[0].num_qubits
        gamma_params = [
            Parameter(f"gamma_{i}_{j}_{r}")
            for r in range(self.params["repetition"])
            for i, j in selected_graphs[0].edge_list()
        ]
        beta_params = [
            Parameter(f"beta_{i}_{r}")
            for r in range(self.params["repetition"])
            for i in selected_graphs[0].node_indexes()
        ]

        self.ansatz = multi_angle_qaoa_circuit(
            gamma_params=gamma_params,
            beta_params=beta_params,
            num_qubits=num_qubits,
            G=selected_graphs[0],
            reps=self.params["repetition"],
        )

        logging.info("number of qubits: %d", num_qubits)
        logging.info("pauli terms: %d", len(pauli_ops[0].paulis))
