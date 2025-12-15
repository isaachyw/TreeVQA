"""
TreeVQA Yielder - Convergence monitoring and operator splitting.

This module provides callbacks for monitoring operator convergence during
VQE optimization and triggering automatic splits when divergence is detected.
"""

import logging
from typing import List, Dict, Tuple, Optional, NoReturn, Callable

import numpy as np
from collections import defaultdict
from sklearn.cluster import SpectralClustering
from scipy.optimize import OptimizeResult
from qiskit.quantum_info import Pauli

from ..treevqa_helper import get_per_sparse_value
from ..op_task import OpTask


class TreeVQA_Yielder:
    """Yielder that monitors operator convergence and triggers splitting.

    Tracks the energy trajectory of each operator and determines when
    operators should be split into separate optimization groups based
    on convergence behavior.

    Attributes:
        window: Length of the consecutive window for slope calculation.
        _ops: List of operators being monitored.
        _values: History of expectation values per operator.
        _warmup: Number of iterations before checking for separation.
    """

    def __init__(
        self,
        window: int,
        operators: List[OpTask],
        warmup: int,
        threshold: Tuple[float, float],
    ) -> None:
        """Initialize TreeVQA Yielder.

        Args:
            window: Length of the consecutive window for slope calculation.
            operators: The Hamiltonian operators to monitor.
            warmup: Number of iterations before checking for separation.
            threshold: (relative_threshold, absolute_threshold) for separation.

        Raises:
            ValueError: If warmup is non-zero and less than window.
        """
        if warmup != 0 and warmup < window:
            raise ValueError("warmup must be >= window")

        self.window = window
        self._ops = operators.copy()
        self._values: Dict[OpTask, List[float]] = defaultdict(list)
        self._warmup = warmup
        self._relative_threshold, self._absolute_threshold = threshold
        self._similarity_matrix = np.zeros((len(self._ops), len(self._ops)))
        self._pre_compute_similarity()

    def __call__(
        self, parameters, pauli_values: Dict[Pauli, float], iter_num
    ) -> Tuple[bool, Optional[List[OpTask]], Optional[List[OpTask]]]:
        """Check if optimization should split into separate operator groups.

        Args:
            parameters: Current parameters (unused, for API compatibility).
            pauli_values: Dictionary of Pauli expectation values.
            iter_num: Current iteration number.

        Returns:
            Tuple of (should_split, group1, group2). If should_split is False,
            groups are None.
        """
        if len(self._ops) <= 1:
            return False, None, None

        self._record_operator_values(pauli_values)

        if len(self._values[self._ops[0]]) > self._warmup:
            slopes = self._compute_slopes()
            if self.should_separate(slopes):
                return True, *self._cluster_operators()

        return False, None, None

    def _record_operator_values(self, pauli_values: Dict[Pauli, float]) -> None:
        """Record the expectation values for each operator.

        Args:
            pauli_values: Dictionary mapping Pauli terms to expectation values.
        """
        for op in self._ops:
            ev = get_per_sparse_value(op, pauli_values)
            self._values[op].append(ev)

    def _compute_slopes(self) -> List[float]:
        """Compute linear regression slopes for each operator over the window.

        Returns:
            List of slopes, one per operator.
        """
        slopes = []
        x_range = np.arange(self.window)
        for op in self._ops:
            last_values = self._values[op][-self.window :]
            slope, _ = np.polyfit(x_range, last_values, 1)
            slopes.append(slope)
        return slopes

    def should_separate(self, slopes: List[float]) -> bool:
        """Determine if operators should be separated based on slope thresholds.

        Args:
            slopes: List of slopes for each operator.

        Returns:
            True if separation criteria are met.
        """
        if len(slopes) <= 1:
            return False

        slopes_arr = np.asarray(slopes)
        if np.any(slopes_arr > self._relative_threshold):
            return True

        mean_slope = np.mean(slopes_arr)
        if mean_slope > self._absolute_threshold:
            logging.info("Separating at mean slope: %s", mean_slope)
            return True

        return False

    def _compute_gaussian_similarity(
        self,
        coeffs_i: np.ndarray,
        coeffs_j: np.ndarray,
        sigma: float = 1.0,
        ord: int = 1,
    ) -> float:
        """Compute Gaussian kernel similarity between two operator coefficient vectors.

        Args:
            coeffs_i: Coefficients of operator i.
            coeffs_j: Coefficients of operator j.
            sigma: Gaussian kernel bandwidth.
            ord: Norm order for distance calculation.

        Returns:
            Gaussian kernel similarity value in [0, 1].
        """
        a_i = np.real(coeffs_i).astype(np.float64)
        a_j = np.real(coeffs_j).astype(np.float64)
        distance = np.linalg.norm(a_i - a_j, ord=ord)
        return np.exp(-(distance**2) / (2 * sigma**2))

    def _pre_compute_similarity(self) -> None:
        """Pre-compute the upper triangular similarity matrix for all operators."""
        n = len(self._ops)
        for i in range(n):
            for j in range(i + 1, n):
                self._similarity_matrix[i, j] = self._compute_gaussian_similarity(
                    self._ops[i].coeffs, self._ops[j].coeffs
                )

    def _cluster_operators(self) -> Tuple[List[OpTask], List[OpTask]]:
        """Cluster operators into two groups using spectral clustering.

        Returns:
            Two groups of operators based on coefficient similarity.
        """
        n = len(self._ops)

        # Trivial case: exactly 2 operators
        if n == 2:
            return [self._ops[0]], [self._ops[1]]

        # Build symmetric affinity matrix from upper triangular similarity
        affinity = self._similarity_matrix + self._similarity_matrix.T
        np.fill_diagonal(affinity, 1.0)

        # Normalize to [0, 1] range and apply spectral clustering
        clustering = SpectralClustering(
            n_clusters=2,
            affinity="precomputed",
            assign_labels="discretize",
            random_state=42,
        )
        labels = clustering.fit_predict(0.5 * (affinity + 1))

        # Partition operators by cluster label
        group1 = [op for op, label in zip(self._ops, labels) if label == 0]
        group2 = [op for op, label in zip(self._ops, labels) if label == 1]

        # Fallback to even split if clustering fails
        if not group1 or not group2:
            mid = n // 2
            group1, group2 = self._ops[:mid], self._ops[mid:]

        logging.info(
            "Spectral clustering: group1=%d ops %s, group2=%d ops %s",
            len(group1),
            [op.op_id for op in group1],
            len(group2),
            [op.op_id for op in group2],
        )
        return group1, group2


class SCIPY_YIELDER(TreeVQA_Yielder):
    """Scipy-compatible callback for TreeVQA optimization with early stopping.

    Extends TreeVQA_Yielder to work as a scipy callback, raising StopIteration
    when a split is needed or max iterations are reached.

    Attributes:
        fun_pauli: Function to get Pauli expectation values from parameters.
        maxiter: Maximum number of iterations before forced stop.
        group1: First operator group after split (None if no split).
        group2: Second operator group after split (None if no split).
        x_history: History of parameter values.
    """

    def __init__(
        self,
        *args,
        fun_pauli: Callable[[np.ndarray], Dict[Pauli, float]],
        maxiter: int,
        **kwargs,
    ) -> None:
        """Initialize scipy-compatible yielder.

        Args:
            fun_pauli: Function to get Pauli expectation values from parameters.
            maxiter: Maximum number of iterations before forced stop.
            *args: Passed to parent TreeVQA_Yielder class.
            **kwargs: Passed to parent TreeVQA_Yielder class.
        """
        super().__init__(*args, **kwargs)
        self.fun_pauli = fun_pauli
        self.maxiter = maxiter
        self.group1: Optional[List[OpTask]] = None
        self.group2: Optional[List[OpTask]] = None
        self.x_history: List[np.ndarray] = []
        self.batch_size = 0

    def __call__(self, intermediate_result: OptimizeResult) -> None | NoReturn:
        """Scipy callback that checks for separation or max iterations.

        Args:
            intermediate_result: Scipy optimization intermediate result.

        Raises:
            StopIteration: When split is triggered or max iterations reached.
        """
        self.x_history.append(intermediate_result.x)
        self.batch_size = intermediate_result.nit

        pauli_values = self.fun_pauli(intermediate_result.x)
        self._record_operator_values(pauli_values)

        if len(self._values[self._ops[0]]) > self._warmup:
            slopes = self._compute_slopes()
            if self.should_separate(slopes):
                self.group1, self.group2 = self._cluster_operators()
                raise StopIteration

        if intermediate_result.nit >= self.maxiter:
            raise StopIteration

        return None
