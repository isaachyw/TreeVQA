# Solve Ising Model with TreeVQA
"""Ising model application for TreeVQA experiments."""

from typing import Dict, List
import numpy as np
import logging
import json

from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import EfficientSU2

from .treevqa_application import TreeVQAApplication
from .Ising_model import load_sparseOp_with_id, load_sparseOp_heisenberg_id
from ..op_task import OpTask
from ..treevqa_helper import average_pauli_hamiltonians
from .util import parse_slice, concatenate_slices


class IsingModelExperiment(TreeVQAApplication):
    """TreeVQA experiment implementation for Ising model systems"""

    DEFAULT_PARAMS = {
        **TreeVQAApplication.DEFAULT_PARAMS,
        "num_node": 2,
        "magnetic_fields": np.linspace(0.0, 1.0, num=8),
    }

    def __init__(self, params: Dict):
        super().__init__(params)
        self.fig_params = {
            "x_label": "Magnetic Field Strength",
            "title": f"Ising Model ({self.params['num_node']} nodes) Energy Landscape",
        }

    @property
    def result_dir_base(self) -> str:
        if self.params["model_type"] == "ising":
            return f"{self.params['result_dir']}/IsingModel{self.params['num_node']}node/{self.params['magnetic_fields']}"
        elif self.params["model_type"] == "heisenberg":
            return f"{self.params['result_dir']}/xxz{self.params['num_node']}node/{self.params['magnetic_fields']}"
        else:
            raise ValueError(f"Invalid model type: {self.params['model_type']}")

    def load_data(self) -> None:
        selected_slices = parse_slice(self.params["magnetic_fields"], precision=3)
        self.selected_magnetic_fields = concatenate_slices(selected_slices)
        logging.info(
            "Selected %d magnetic fields: %s",
            len(self.selected_magnetic_fields),
            self.selected_magnetic_fields,
        )
        if self.params["model_type"] == "ising":
            op_gen = load_sparseOp_with_id(
                self.selected_magnetic_fields, self.params["num_node"]
            )
        elif self.params["model_type"] == "heisenberg":
            op_gen = load_sparseOp_heisenberg_id(
                self.selected_magnetic_fields, self.params["num_node"]
            )
        magnetic_fields, pauli_ops = zip(*op_gen)
        magnetic_fields, pauli_ops = list(magnetic_fields), list(pauli_ops)

        op_task_proxy = [OpTask(op, op_id=0) for op in pauli_ops]
        pauli_strs = [op.hinfo for op in op_task_proxy]

        self.avg_hinfo, processed_pauli_strs = average_pauli_hamiltonians(
            pauli_strs, np.mean
        )
        pauli_ops = [
            SparsePauliOp(data=hinfo[1], coeffs=hinfo[0])
            for hinfo in processed_pauli_strs
        ]

        if self.params["model_type"] == "ising":
            json_file = f"ground-state/LL-{self.params['num_node']}.json"
        else:
            json_file = f"ground-state/XXZ-re-{self.params['num_node']}.json"
        with open(json_file, "r", encoding="utf-8") as f:
            energy_map = json.load(f)
            energy_map = {float(k): v for k, v in energy_map.items()}
            try:
                self.ref_values = [
                    energy_map[bond_length]
                    for bond_length in self.selected_magnetic_fields
                ]
            except KeyError:
                raise ValueError("Reference values not found in JSON file.")
        logging.info("Reference values: %s", [round(val, 5) for val in self.ref_values])
        self.op_tasks = [
            OpTask(op, op_id=mag) for op, mag in zip(pauli_ops, magnetic_fields)
        ]

        num_qubits = self.op_tasks[0].num_qubits
        self.ansatz = EfficientSU2(
            num_qubits, entanglement="circular", reps=self.params["repetition"]
        )
        logging.info("number of qubits: %d", num_qubits)
        logging.info("pauli terms : %d", len(pauli_ops[0].paulis))
