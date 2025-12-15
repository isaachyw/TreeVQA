"""
SPSA optimizer with TreeVQA support.

This module extends SPSA with automatic operator splitting capabilities
for parallel VQE optimization.
"""

import warnings
import logging
from time import time
from dataclasses import dataclass
from collections import deque
from typing import (
    Iterator,
    Callable,
    SupportsFloat,
    Tuple,
    Optional,
    Union,
    Generator,
    List,
)

import numpy as np
from qiskit_algorithms.optimizers import SPSA, OptimizerResult
from qiskit_algorithms.optimizers.spsa import (
    _validate_pert_and_learningrate,
    bernoulli_perturbation,
    _batch_evaluate,
)

from ..op_task import OpTask

# =============================================================================
# Type Aliases
# =============================================================================

CALLBACK = Callable[[int, np.ndarray, float, SupportsFloat, bool], None]
TC_YIELD = Callable[
    [int, np.ndarray, float, SupportsFloat, bool], Tuple[bool, Optional[OpTask]]
]
POINT = Union[float, np.ndarray]

# Calibration constants
CALIBRATION_STEPS = 25


@dataclass
class SPSAHyperParams:
    """Hyperparameters for SPSA learning rate and perturbation schedules.

    Attributes:
        a: Learning rate coefficient.
        alpha: Learning rate decay exponent.
        stability_constant: Stability constant A for learning rate schedule.
        c: Perturbation coefficient.
        gamma: Perturbation decay exponent.
    """

    a: float
    alpha: float
    stability_constant: float
    c: float
    gamma: float


class SPSAP(SPSA):
    """SPSA optimizer with TreeVQA (parallel) support.

    Extends SPSA with convergence monitoring and automatic operator
    splitting via the TreeVQA yielder callback.

    Attributes:
        cap_yielder: Callback for checking convergence and triggering splits.
        k_offset: Iteration offset for continuation (used after splits).
        eta: Learning rate generator.
        eps: Perturbation generator.
    """

    def __init__(
        self,
        maxiter: int = 100,
        blocking: bool = False,
        allowed_increase: float | None = None,
        trust_region: bool = False,
        learning_rate: float | np.ndarray | Callable[[], Iterator] | None = None,
        perturbation: float | np.ndarray | Callable[[], Iterator] | None = None,
        last_avg: int = 1,
        resamplings: int | dict[int, int] = 1,
        perturbation_dims: int | None = None,
        second_order: bool = False,
        regularization: float | None = None,
        hessian_delay: int = 0,
        lse_solver: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
        initial_hessian: np.ndarray | None = None,
        callback: CALLBACK | None = lambda *args, **kwargs: None,
        cap_yielder: TC_YIELD | None = None,
    ) -> None:
        """Initialize SPSAP optimizer.

        Args:
            cap_yielder: Callback for checking convergence and triggering splits.
            Other args are passed to parent SPSA class.
        """
        super().__init__(
            maxiter=maxiter,
            blocking=blocking,
            allowed_increase=allowed_increase,
            trust_region=trust_region,
            learning_rate=learning_rate,
            perturbation=perturbation,
            last_avg=last_avg,
            resamplings=resamplings,
            perturbation_dims=perturbation_dims,
            second_order=second_order,
            regularization=regularization,
            hessian_delay=hessian_delay,
            lse_solver=lse_solver,
            initial_hessian=initial_hessian,
            callback=callback,
            termination_checker=None,
        )
        self.cap_yielder = cap_yielder
        self.k_offset = 0
        if learning_rate is None and perturbation is None:
            logging.info(
                "dummy spsa is create to calibrate the learning rate and perturbation"
            )
        else:
            self._init_learning_schedules(learning_rate, perturbation)

    def _init_learning_schedules(
        self,
        learning_rate: float | np.ndarray | Callable[[], Iterator] | None,
        perturbation: float | np.ndarray | Callable[[], Iterator] | None,
    ) -> None:
        """Initialize learning rate (eta) and perturbation (eps) generators.

        If both learning_rate and perturbation are None, initialization is
        deferred until calibration (eta/eps will be None). They must be set
        via reinitializing the optimizer after calibration.
        """
        get_eta, get_eps = _validate_pert_and_learningrate(perturbation, learning_rate)
        if isinstance(
            get_eta, Generator
        ):  # use the old generator passed in by parent vqe cluster
            self.eta, self.eps = get_eta, get_eps
        else:
            self.eta, self.eps = get_eta(), get_eps()

    def minimize(
        self,
        fun: Callable[[POINT], float],
        x0: POINT,
        jac: Callable[[POINT], POINT] | None = None,
        bounds: list[tuple[float, float]] | None = None,
        fun_pauli: Callable[[POINT], Tuple[float, dict]] | None = None,
    ) -> Tuple[OptimizerResult, Optional[List[OpTask]], Optional[List[OpTask]]]:
        """Run SPSA optimization with TreeVQA support.

        Args:
            fun: Objective function to minimize.
            x0: Initial parameters.
            jac: Jacobian function (unused, for API compatibility).
            bounds: Parameter bounds (unused, for API compatibility).
            fun_pauli: Function returning (energy, pauli_values) for TreeVQA callbacks.

        Returns:
            Tuple of (optimizer_result, group1, group2). Groups are set if
            the yielder triggered a split.

        Raises:
            ValueError: If learning rate (eta) or perturbation (eps) schedules are not initialized.
        """
        x = np.asarray(x0)
        lse_solver = self.lse_solver or np.linalg.solve
        self._smoothed_hessian = self.initial_hessian or np.identity(x.size)
        self._nfev = 0

        logging.info("SPSAP: Starting optimization")
        start = time()

        last_steps = deque([x])
        group1, group2 = None, None

        for k in range(1, self.maxiter + 1):
            _, update = self._compute_update(fun, x, k, next(self.eps), lse_solver)

            if self.trust_region:
                norm = np.linalg.norm(update)
                if norm > 1:
                    update = update / norm

            update *= next(self.eta)
            x_next = x - update

            fx_next, pauli_values = self._evaluate_and_callback(
                fun, fun_pauli, x_next, update
            )

            if k % 10 == 0:
                logging.info("Iteration %d/%d, energy=%.4f", k, self.maxiter, fx_next)

            x = x_next

            if self.last_avg > 1:
                last_steps.append(x_next)
                if len(last_steps) > self.last_avg:
                    last_steps.popleft()

            if self.cap_yielder is not None:
                is_split, group1, group2 = self.cap_yielder(
                    x, pauli_values, k + self.k_offset
                )
                if is_split:
                    logging.info("Split triggered at iteration %d/%d", k, self.maxiter)
                    break

        logging.info("SPSAP: Finished in %.2fs", time() - start)

        if self.last_avg > 1:
            x = np.mean(np.asarray(last_steps), axis=0)

        return self._build_result(fun, x, k), group1, group2

    def _evaluate_and_callback(
        self,
        fun: Callable[[POINT], float],
        fun_pauli: Callable[[POINT], Tuple[float, dict]] | None,
        x: np.ndarray,
        update: np.ndarray,
    ) -> Tuple[float, Optional[dict]]:
        """Evaluate objective and invoke callback if set."""
        fx, pauli_values = None, None

        if self.callback is not None and not self.blocking:
            self._nfev += 1
            if fun_pauli is not None:
                fx, pauli_values = fun_pauli(x)
            else:
                fx = fun(x)

            self.callback(self._nfev, x, fx, np.linalg.norm(update), True)

        return fx, pauli_values

    def _build_result(self, fun: Callable, x: np.ndarray, nit: int) -> OptimizerResult:
        """Construct OptimizerResult from final state."""
        result = OptimizerResult()
        result.x = x
        result.fun = fun(x)
        result.nfev = self._nfev
        result.nit = nit
        return result

    def update_k_offset(self, k_offset: int) -> None:
        """Update iteration offset for TreeVQA yielder (used when resuming optimization).

        Args:
            k_offset: New iteration offset value.
        """
        self.k_offset = k_offset

    @staticmethod
    def get_hyperpara(
        loss: Callable[[np.ndarray], float],
        initial_point: np.ndarray,
        c: float = 0.1,
        stability_constant: float = 0,
        target_magnitude: float | None = None,
        alpha: float = 0.502,
        gamma: float = 0.101,
        modelspace: bool = False,
        max_evals_grouped: int = 1,
    ) -> SPSAHyperParams:
        """Calibrate SPSA hyperparameters by estimating gradient magnitude.

        Args:
            loss: Objective function.
            initial_point: Starting parameters.
            c: Initial perturbation coefficient.
            stability_constant: Stability constant A for learning rate schedule.
            target_magnitude: Target step size magnitude.
            alpha: Learning rate decay exponent.
            gamma: Perturbation decay exponent.
            modelspace: Use squared magnitude for model-based SPSA.
            max_evals_grouped: Batch size for loss evaluations.

        Returns:
            SPSAHyperParams with calibrated values.
        """
        logging.info("SPSAP: Calibrating hyperparameters")

        if target_magnitude is None:
            target_magnitude = np.pi / 8

        dim = len(initial_point)

        points = []
        for _ in range(CALIBRATION_STEPS):
            pert = bernoulli_perturbation(dim)
            points.extend([initial_point + c * pert, initial_point - c * pert])

        losses = _batch_evaluate(loss, points, max_evals_grouped)

        avg_magnitude = (
            sum(
                np.abs((losses[2 * i] - losses[2 * i + 1]) / (2 * c))
                for i in range(CALIBRATION_STEPS)
            )
            / CALIBRATION_STEPS
        )

        a = target_magnitude / (avg_magnitude**2 if modelspace else avg_magnitude)

        if a < 1e-10 or a > 10:
            warnings.warn(f"Calibration failed, using {target_magnitude} for `a`")
            a = target_magnitude

        logging.info(
            "Calibration complete: a=%.4f, A=%.4f, alpha=%.3f, c=%.4f, gamma=%.3f",
            a,
            stability_constant,
            alpha,
            c,
            gamma,
        )

        return SPSAHyperParams(a, alpha, stability_constant, c, gamma)
