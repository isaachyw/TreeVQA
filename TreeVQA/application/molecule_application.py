# Solve molecule ground energy with TreeVQA
"""Molecule application for TreeVQA experiments."""

import numpy as np
import json
from typing import Dict, List
import logging

from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import EfficientSU2
from qiskit_nature.units import DistanceUnit

from .treevqa_application import TreeVQAApplication
from ..op_task import OpTask
from .util import parse_slice, concatenate_slices
from ..treevqa_helper import (
    average_pauli_hamiltonians,
    truncate_pauli_string,
    uccsd_ansatz,
    get_molecule_coords,
    molecule_to_op,
)


class MoleculeExperiment(TreeVQAApplication):
    """TreeVQA experiment implementation for molecular systems"""

    DEFAULT_PARAMS = {
        **TreeVQAApplication.DEFAULT_PARAMS,
        "molecule_name": "H2",
        "molecule_slice": "0:2,-2:",
        "freeze": False,
        "truncate_ratio": 0.005,
    }

    def __init__(self, params: Dict):
        super().__init__(params)
        self.selected_bond_lengths = []
        self.fig_params = {
            "x_label": "Bond Length (Å)",
            "title": f"{self.params['molecule_name']} Potential Energy Surface",
        }

    @property
    def result_dir_base(self) -> str:
        return f"{self.params['result_dir']}/{self.params['molecule_name']}/{self.params['molecule_slice']}"

    @property
    def result_dir(self) -> str:
        """Generate result directory path"""
        return (
            f"{self.result_dir_base}/"
            f"VQE{self.params['vqe_iter']}_"
            f"W{self.params['cap_window']}_"
            f"WA{self.params['cap_warmup']}_"
            f"T{self.formatted_threshold}_"
            f"{self.params['ansatz']}"
        )

    @property
    def formatted_threshold(self) -> str:
        """Format threshold for result directory"""
        return f"{self.params['threshold'][0]}_{self.params['threshold'][1]}"

    def load_data(self) -> None:
        selected_slices = parse_slice(self.params["molecule_slice"], precision=2)
        self.selected_bond_lengths = concatenate_slices(selected_slices)
        logging.info(
            "Selected %d bond lengths: %s",
            len(self.selected_bond_lengths),
            [float(bond_length) for bond_length in self.selected_bond_lengths],
        )

        json_file = (
            f"ground-state/{self.params['molecule_name']}.json"
            if self.params["freeze"]
            else f"ground-state/{self.params['molecule_name']}.json"
        )
        with open(json_file, "r", encoding="utf-8") as f:
            energy_map = json.load(f)
            energy_map = {float(k): v for k, v in energy_map.items()}
            try:
                self.ref_values = [
                    energy_map[bond_length]
                    for bond_length in self.selected_bond_lengths
                ]
            except KeyError:
                raise ValueError("Reference values not found in JSON file.")
        logging.info("Reference values: %s", [round(val, 5) for val in self.ref_values])

        molecule_coords = [
            get_molecule_coords(self.params["molecule_name"], bond)
            for bond in self.selected_bond_lengths
        ]
        Hinfos = [
            molecule_to_op(molecule_coord, DistanceUnit.ANGSTROM, freeze_core=False)
            for molecule_coord in molecule_coords
        ]
        pauli_strs = [(H[0], H[1]) for H in Hinfos]
        self.hf_bits = Hinfos[0][2]
        logging.info("HF bits: %s", self.hf_bits)
        self.avg_hinfo, processed_pauli_strs = average_pauli_hamiltonians(
            pauli_strs, np.mean
        )
        avg_pauli_ops = SparsePauliOp(
            data=self.avg_hinfo[1], coeffs=self.avg_hinfo[0]
        ).sort()
        pauli_ops = [
            SparsePauliOp(data=hinfo[1], coeffs=hinfo[0]).sort()
            for hinfo in processed_pauli_strs
        ]

        self.pauli_ops = truncate_pauli_string(
            avg_pauli_ops.copy(), pauli_ops.copy(), ratio=self.params["truncate_ratio"]
        )

        self.num_qubits = len(processed_pauli_strs[0][1][0])
        if self.params["ansatz"] == "UCCSD":
            self.ansatz = uccsd_ansatz(
                self.params["molecule_name"],
                self.params["freeze"],
                self.params["repetition"],
            ).decompose()
        elif self.params["ansatz"] == "HEA":
            self.ansatz = EfficientSU2(
                self.num_qubits, entanglement="circular", reps=self.params["repetition"]
            )
        else:
            raise ValueError("Invalid ansatz type, can't construct ansatz.")
        logging.info("number of qubits: %d", self.num_qubits)
        logging.info("pauli terms : %d", len(self.pauli_ops[0].paulis))

        self.selected_bond_lengths = [
            float(bond_length) for bond_length in self.selected_bond_lengths
        ]
        self.op_tasks = [
            OpTask(op.paulis, coeffs=op.coeffs, op_id=bond_length)
            for bond_length, op in zip(self.selected_bond_lengths, self.pauli_ops)
        ]
