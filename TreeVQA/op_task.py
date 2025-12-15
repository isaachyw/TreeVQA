"""
OpTask - Operator container for TreeVQA optimization.

This module provides the OpTask class which wraps SparsePauliOp with additional
tracking capabilities for TreeVQA optimization.
"""

from __future__ import annotations

import random
from typing import List, Union, Tuple

import numpy as np
from qiskit.quantum_info import SparsePauliOp


class OpTask(SparsePauliOp):
    """Operator task for TreeVQA with identity tracking via op_id.

    OpTask extends SparsePauliOp to include an operator ID for tracking
    during parallel optimization. Operators with the same Pauli terms
    but different coefficients should have different op_ids.

    Attributes:
        op_id: Unique identifier (int or float for bond length).
               -1 indicates auxiliary/averaged operators.
    """

    def __init__(self, *args, op_id: Union[int, float], **kwargs) -> None:
        """Initialize OpTask with operator ID.

        Args:
            *args: Arguments passed to SparsePauliOp.
            op_id: Unique identifier for this operator.
            **kwargs: Keyword arguments passed to SparsePauliOp.
        """
        super().__init__(*args, **kwargs)
        self.op_id = op_id
        self._imutable_repr = self.op_id
        self._hash = hash(self.op_id)

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OpTask):
            return False
        return self.op_id == other.op_id

    def __repr__(self) -> str:
        return f"OpTask({self.op_id})"

    def __copy__(self) -> OpTask:
        """Create a shallow copy of this OpTask object."""
        return OpTask(self.paulis, coeffs=self.coeffs, op_id=self.op_id)

    @property
    def hinfo(self) -> Tuple[np.ndarray, list]:
        """Return (coefficients, paulis) tuple."""
        return self.coeffs, self.paulis

    # =========================================================================
    # Static Analysis Methods
    # =========================================================================

    @staticmethod
    def design_matrix(op_tasks: List[OpTask]) -> np.ndarray:
        """Build design matrix from operator coefficients.

        Excludes identity terms and returns coefficient matrix.

        Args:
            op_tasks: List of OpTask operators.

        Returns:
            2D array of shape (n_operators, n_non_identity_terms).
        """
        num_qubits = op_tasks[0].num_qubits
        identity_str = "I" * num_qubits
        arr = np.array(
            [
                [c for c, p in zip(op.coeffs, op.paulis) if p != identity_str]
                for op in op_tasks
            ],
            dtype=np.float64,
        )
        return arr

    @staticmethod
    def normalized_variance(op_tasks: List[OpTask]) -> float:
        """Compute normalized variance across operators.

        Args:
            op_tasks: List of OpTask operators with same Pauli terms.

        Returns:
            Normalized variance in [0, 1].
        """
        design_matrix = OpTask.design_matrix(op_tasks)
        return OpTask.compute_variance(design_matrix)

    @staticmethod
    def compute_variance(arr: np.ndarray) -> float:
        """Compute normalized variance of a 2D array.

        Normalizes by maximum possible variance for the data range.

        Args:
            arr: 2D numpy array.

        Returns:
            Normalized variance in [0, 1].
        """
        variance = np.var(arr)
        range_val = np.max(arr) - np.min(arr)

        if range_val == 0:
            return 0.0

        max_variance = (range_val / 2) ** 2
        return variance / max_variance

    @staticmethod
    def group_variance(op_tasks: List[OpTask], group_size: int) -> float:
        """Compute mean normalized variance across random groups.

        Args:
            op_tasks: List of OpTask operators.
            group_size: Size of random groups.

        Returns:
            Mean normalized variance across groups.
        """
        design_matrix = OpTask.design_matrix(op_tasks)
        return mean_group_normalized_variance(design_matrix, group_size=group_size)


class HashableSparsePauliOp:
    """Hashable wrapper for SparsePauliOp.

    Allows SparsePauliOp to be used in sets and as dictionary keys.
    """

    def __init__(self, op: SparsePauliOp) -> None:
        """Initialize wrapper.

        Args:
            op: SparsePauliOp to wrap.
        """
        self.op = op
        self._immutable_repr = self._compute_immutable_repr()
        self._hash = hash(self._immutable_repr)

    def _compute_immutable_repr(self) -> tuple:
        """Compute hashable representation."""
        labels = self.op.paulis.to_labels()
        coeffs = [complex(c) for c in self.op.coeffs]
        return tuple(zip(labels, coeffs))

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HashableSparsePauliOp):
            return False
        return self._immutable_repr == other._immutable_repr

    def __repr__(self) -> str:
        return f"HashableSparsePauliOp({self.op})"


def mean_group_normalized_variance(data: np.ndarray, group_size: int = 4) -> float:
    """Compute mean normalized variance across random groups.

    Randomly shuffles data and computes normalized variance for each
    group of specified size, then returns the mean.

    Args:
        data: Input array to group and analyze.
        group_size: Number of elements per group.

    Returns:
        Mean normalized variance across all groups.

    Raises:
        ValueError: If data is empty.
    """
    n = len(data)
    if n == 0:
        raise ValueError("Data must contain at least one element.")

    shuffled = data.copy()
    random.shuffle(shuffled)

    groups = [shuffled[i : i + group_size] for i in range(0, n, group_size)]

    normalized_variances = [
        OpTask.compute_variance(np.array(group)) for group in groups
    ]

    return np.nanmean(normalized_variances)


if __name__ == "__main__":
    # Simple test
    op = OpTask(["II", "XI"], coeffs=[2.0, 1.2], op_id=0)
    print(op)
    print(f"op_id: {op.op_id}")
    print(f"coeffs: {op.coeffs}")
    print(f"paulis: {op.paulis}")
