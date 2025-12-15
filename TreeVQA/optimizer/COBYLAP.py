"""
COBYLA optimizer with TreeVQA support.

This module extends COBYLA with automatic operator splitting capabilities
for parallel VQE optimization.
"""

import time
import logging
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List

import numpy as np
from scipy.optimize import minimize
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.optimizers.optimizer import OptimizerResult

from .TreeVQA_Yielder import SCIPY_YIELDER
from ..op_task import OpTask


@dataclass
class COBYLAPConfig:
    """Hyperparameters for COBYLAP optimizer.

    Attributes:
        rhobeg: Initial trust region radius.
    """

    rhobeg: Optional[float] = None


class COBYLAP(COBYLA):
    """COBYLA optimizer with TreeVQA (parallel) support.

    Extends COBYLA with convergence monitoring and automatic operator
    splitting via the TreeVQA yielder callback.

    Attributes:
        cap_yielder: Callback for checking convergence and triggering splits.
        k_offset: Iteration offset for continuation (unused in COBYLA).
    """

    def __init__(
        self,
        *args,
        cap_yielder: SCIPY_YIELDER,
        optimize_config: dict,
        **kwargs,
    ) -> None:
        """Initialize COBYLAP optimizer.

        Args:
            cap_yielder: Scipy-compatible yielder for convergence checking.
            optimize_config: Dictionary with optimizer settings (e.g., rhobeg).
            *args: Passed to parent COBYLA class.
            **kwargs: Passed to parent COBYLA class.
        """
        super().__init__(*args, **kwargs)
        self.cap_yielder = cap_yielder
        self.k_offset = 0
        self._config = COBYLAPConfig(**optimize_config)

    def minimize(
        self, fun, x0
    ) -> Tuple[OptimizerResult, Optional[List[OpTask]], Optional[List[OpTask]]]:
        """Run COBYLA optimization with TreeVQA callback.

        Args:
            fun: Objective function to minimize.
            x0: Initial parameters.

        Returns:
            Tuple of (optimizer_result, group1, group2). Groups are set if
            the yielder triggered a split.
        """
        initial_len = len(self.cap_yielder._values.get(self.cap_yielder._ops[0], []))

        if self._config.rhobeg is None:
            self._config.rhobeg = np.pi

        logging.info("COBYLAP options: %s", asdict(self._config))

        start_time = time.time()
        scipy_result = minimize(
            fun,
            x0,
            method="cobyla",
            callback=self.cap_yielder,
            options=asdict(self._config),
        )
        elapsed = time.time() - start_time

        logging.info(
            "COBYLAP completed in %.2fs, energy=%.6f", elapsed, scipy_result.fun
        )

        result = self._to_optimizer_result(scipy_result, initial_len)

        # Update rhobeg for potential continuation
        self._config.rhobeg = self.rhobeg

        self._log_termination_status(scipy_result.status)

        return result, self.cap_yielder.group1, self.cap_yielder.group2

    def _to_optimizer_result(self, scipy_result, initial_len: int) -> OptimizerResult:
        """Convert scipy OptimizeResult to qiskit OptimizerResult.

        Args:
            scipy_result: Scipy optimization result.
            initial_len: Initial value history length for iteration counting.

        Returns:
            Qiskit OptimizerResult.
        """
        result = OptimizerResult()
        result.x = scipy_result.x
        result.fun = scipy_result.fun
        result.nfev = scipy_result.nfev
        result.njev = scipy_result.get("njev", None)

        final_len = len(self.cap_yielder._values[self.cap_yielder._ops[0]])
        result.nit = final_len - initial_len
        return result

    def _log_termination_status(self, status: int) -> None:
        """Log appropriate message based on scipy termination status.

        Args:
            status: Scipy termination status code.
        """
        if status == 30:
            logging.info("COBYLAP: StopIteration raised (splitting)")
        elif status in (0, 3):
            logging.info("COBYLAP: Normal termination (status=%d)", status)
            logging.info(
                "Value history length: %d",
                len(self.cap_yielder._values[self.cap_yielder._ops[0]]),
            )
        else:
            logging.warning("COBYLAP: Unexpected termination status=%d", status)

    @property
    def rhobeg(self) -> float:
        """Estimate rhobeg from recent parameter step size.

        Returns:
            Trust region radius estimate (minimum 1e-3).
        """
        if len(self.cap_yielder.x_history) < 2:
            return self._config.rhobeg or np.pi

        step_size = np.linalg.norm(
            self.cap_yielder.x_history[-1] - self.cap_yielder.x_history[-2]
        )
        return max(1e-3, step_size)
