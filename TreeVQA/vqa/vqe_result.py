"""
VQE result containers for TreeVQA optimization.

This module provides data structures for storing VQE optimization results
including energy trajectories and operator histories.
"""

from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import scienceplots  # noqa: F401 (required for plt.style.context)
import matplotlib.pyplot as plt
from qiskit.quantum_info import Pauli

from ..op_task import OpTask


@dataclass
class SegmentEnergy:
    """Energy values for an operator over a time segment.

    Attributes:
        start: Starting iteration index.
        end: Ending iteration index (exclusive).
        energy: List of energy values for each iteration.
    """

    start: int
    end: int
    energy: List[float]


@dataclass
class VQEClusterResult:
    """Result container for VQECluster optimization run.

    Stores the complete history of a VQE optimization including Pauli
    expectation values, operator groupings, and split events.

    Attributes:
        vid: VQE identifier.
        starting_step: Initial step when this VQE was created.
        pauli_expectation: History of Pauli expectation values per iteration.
        final_ops: Operators remaining at the end of optimization.
        bucket_ids: History of (ending_step, operators) tuples tracking splits.
        actual_step: Total cumulative steps completed.
    """

    vid: int | float
    starting_step: int
    pauli_expectation: Dict[Pauli, List[float]]
    final_ops: List[OpTask]
    bucket_ids: List[Tuple[int, List[OpTask]]]
    actual_step: int

    @property
    def final_energy(self) -> List[float]:
        """Get final energy of each operator in the bucket.

        Returns:
            List of final energies, one per operator.
        """
        if not self.final_ops:
            return []

        pauli_terms = (
            self.final_ops[0].paulis
            if self.final_ops
            else self.pauli_expectation.keys()
        )

        final_pauli_exp = [
            self.pauli_expectation[Pauli(pauli)][-1] for pauli in pauli_terms
        ]

        return [np.dot(final_pauli_exp, op.coeffs.real) for op in self.final_ops]

    def plot_energys(
        self, fig_dir: str, is_draw: bool = True
    ) -> Dict[int, List[SegmentEnergy]]:
        """Plot energy trajectories for all operators across segments.

        Args:
            fig_dir: Directory to save the figure.
            is_draw: If True, generate and save the plot.

        Returns:
            Dictionary mapping operator IDs to their energy segments.
        """
        all_ops_energy = self._recover_all_energy()
        energy_by_op: Dict[int, List[SegmentEnergy]] = defaultdict(list)

        all_op_ids = {op.op_id for _, ops in self.bucket_ids for op in ops}
        colormap = plt.cm.get_cmap("tab10")
        color_map = {
            op_id: colormap(i % colormap.N) for i, op_id in enumerate(all_op_ids)
        }

        labeled_ops: Set[int] = set()

        with plt.style.context(("science", "grid")):
            last_ending = self.starting_step

            for ending_step, ops in self.bucket_ids:
                time_segment = np.arange(last_ending, ending_step)
                slice_start = last_ending - self.starting_step
                slice_end = ending_step - self.starting_step

                for op in ops:
                    energy_segment = all_ops_energy[op][slice_start:slice_end]
                    energy_by_op[op.op_id].append(
                        SegmentEnergy(
                            start=last_ending, end=ending_step, energy=energy_segment
                        )
                    )

                    if is_draw:
                        label = f"{op.op_id}" if op.op_id not in labeled_ops else None
                        plt.plot(
                            time_segment,
                            energy_segment,
                            label=label,
                            color=color_map[op.op_id],
                        )
                        labeled_ops.add(op.op_id)

                last_ending = ending_step

            if is_draw:
                plt.xlabel("Iteration")
                plt.ylabel("Energy")
                plt.title(f"VQE{self.vid}")
                plt.legend(prop={"size": 6, "weight": "bold"})
                plt.savefig(f"{fig_dir}/{self.vid}.png", dpi=250)
                plt.close()

        return energy_by_op

    def _recover_op_energy(self, op: OpTask) -> np.ndarray:
        """Recover energy time series for a single operator.

        Args:
            op: The operator to recover energy for.

        Returns:
            1D array of energy values over time.
        """
        exp_array = np.array(
            [
                np.array(self.pauli_expectation[Pauli(pauli)]) * coeff.real
                for pauli, coeff in zip(op.paulis, op.coeffs)
            ]
        )
        return np.sum(exp_array, axis=0)

    def _recover_all_energy(self) -> Dict[OpTask, np.ndarray]:
        """Recover energy time series for all operators that appeared.

        Returns:
            Dictionary mapping each operator to its energy time series.
        """
        all_ops = {op for _, ops in self.bucket_ids for op in ops}
        return {op: self._recover_op_energy(op) for op in all_ops}
