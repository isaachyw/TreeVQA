"""
VQE implementation with TreeVQA support.

This module extends Qiskit's VQE to support TreeVQA optimization.
"""

from time import time
from copy import deepcopy
from typing import Callable, Tuple, Optional, List, Dict, Any
import logging
from collections import defaultdict

import numpy as np
from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.quantum_info import Pauli
from qiskit_aer import AerSimulator
from qiskit_algorithms.optimizers import Optimizer
from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit_algorithms.exceptions import AlgorithmError
from qiskit_algorithms.minimum_eigensolvers.vqe import (
    _set_default_batchsize,
    validate_bounds,
    validate_initial_point,
)

from .pp_util import create_efficientsu2_circuit, estimate_pp_per_pauli, estimate_pp
from .vqe_result import VQEClusterResult
from ..op_task import OpTask
from ..optimizer.SPSAP import SPSAP, SPSAHyperParams
from ..optimizer.COBYLAP import COBYLAP
from ..optimizer.TreeVQA_Yielder import TreeVQA_Yielder, SCIPY_YIELDER
from ..noise_config import build_device_noise_model


# =============================================================================
# Configuration Constants
# =============================================================================

DEVICE = "CPU"  # GPU support disabled for stability
JULIA_THRESHOLD = 20  # Use Julia-based PauliPropagation for large circuits


class VQECluster(VQE):
    """VQE optimizer with TreeVQA support.

    Extends the standard VQE to monitor operator convergence and dynamically
    split operators into separate optimization groups when divergence is detected.

    Attributes:
        vid: VQE instance identifier.
        parrent_id: Parent VQE ID (-1 for root).
        starting_step: Step count when this VQE was created.
        culmulative_step: Total steps from root.
        step_size: Steps per optimization batch.
        budget: Maximum steps (None for unlimited).
        cap_window_size: Window size for convergence monitoring.
        cap_warmup: Warmup iterations before convergence checks.
    """

    def __init__(
        self,
        estimator,
        ansatz: QuantumCircuit,
        ops: List[OpTask],
        averager: Callable[[List[OpTask]], OpTask],
        vid: int,
        parrent_id: int,
        optimizer: Optimizer,
        starting_step: int,
        budget: Optional[int],
        *,
        noisy: bool = False,
        culmulative_step: int,
        cap_window_size: int,
        cap_warmup: int,
        yielder_threshold: float | Tuple[float, float],
        step_size: int,
        gradient=None,
        optimizer_method: str = "SPSAP",
        optimizer_configs: Optional[Dict[str, Any]] = None,
        initial_point: np.ndarray | None = None,
        callback: Callable[[int, np.ndarray, float, dict[str, Any]], None]
        | None = None,
        np_log_name: str | None = None,
        gate_error: float = 0.01,
    ) -> None:
        """Initialize VQECluster optimizer.

        Args:
            estimator: Qiskit estimator primitive.
            ansatz: Parameterized quantum circuit.
            ops: List of operators to optimize.
            averager: Function to combine operators for joint optimization.
            vid: VQE identifier.
            parrent_id: Parent VQE ID (-1 for root).
            optimizer: Optimizer instance or class.
            starting_step: Initial step count.
            budget: Maximum steps (None for unlimited).
            noisy: Enable noise simulation.
            culmulative_step: Total steps from root.
            cap_window_size: Window size for convergence monitoring.
            cap_warmup: Warmup iterations before convergence checks.
            yielder_threshold: Threshold(s) for triggering splits.
            step_size: Steps per optimization batch.
            gradient: Gradient function (optional).
            optimizer_method: "SPSAP" or "COBYLAP".
            optimizer_configs: Additional optimizer configuration.
            initial_point: Starting parameters.
            callback: Iteration callback function.
            np_log_name: Log file name.
            gate_error: Gate error rate for noise model.
        """
        super().__init__(
            estimator=estimator,
            ansatz=ansatz,
            optimizer=optimizer,
            gradient=gradient,
            initial_point=initial_point,
            callback=callback,
        )

        # Core state
        self._ops = ops
        self._target_op = averager(ops)
        self._active = True
        self._k = 0  # Internal iteration counter

        # Configuration
        self.vid = vid
        self.parrent_id = parrent_id
        self.starting_step = starting_step
        self.culmulative_step = culmulative_step
        self.step_size = step_size
        self.budget = budget
        self.averager = averager

        # TreeVQA parameters
        self.cap_window_size = cap_window_size
        self.cap_warmup = cap_warmup
        self.yielder_threshold = yielder_threshold

        # Optimizer configuration
        self.optimizer_method = optimizer_method
        self.optimizer_configs = optimizer_configs or {}
        self.paras = validate_initial_point(initial_point, self.ansatz)

        # History tracking
        self.op_history: Dict[Pauli, List[float]] = defaultdict(list)
        self.bucket_ids: List[Tuple[int, List[OpTask]]] = []
        self.been_shortened = False
        self.np_log_name = np_log_name

        # Noise configuration
        self.noisy = noisy
        self.noise_model = build_device_noise_model("FakeKolkataV2") if noisy else None

        # Initialize optimizer
        self.optimizer = self._calibrate_optimizer(optimizer)

        logging.info(
            "VQE%d initialized: warmup=%d, window=%d",
            self.vid,
            self.cap_warmup,
            self.cap_window_size,
        )
        if ansatz.num_qubits > JULIA_THRESHOLD:
            logging.info("Using Julia-based PauliPropagation for large circuit")

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def ops(self) -> List[OpTask]:
        """Current operators being optimized."""
        return self._ops

    @property
    def is_active(self) -> bool:
        """Whether this VQE instance is still running."""
        return self._active

    @is_active.setter
    def is_active(self, active: bool) -> None:
        self._active = active
        self._update_buckets()

    @property
    def individual_step(self) -> int:
        """Steps elapsed since this VQE was spawned."""
        return self.culmulative_step - self.starting_step

    # =========================================================================
    # Operator Management
    # =========================================================================

    def _update_buckets(self) -> None:
        """Record current operators state to bucket history."""
        self.bucket_ids.append((self.culmulative_step, self._ops.copy()))

    def _refresh_after_op_change(self) -> None:
        """Reset optimizer state after operators change."""
        self._k = 0
        self._target_op = self.averager(self._ops)
        self._reset_optimizer()

    def add_op(self, op: OpTask) -> None:
        """Add an operator and refresh optimizer."""
        self._update_buckets()
        self._ops.append(op)
        self._refresh_after_op_change()

    def remove_op(self, op: OpTask) -> None:
        """Remove an operator; deactivate if no operators remain."""
        self._update_buckets()
        self._ops.remove(op)
        if not self._ops:
            self._active = False
        else:
            self._refresh_after_op_change()

    def remove_multiple_ops(self, ops: List[OpTask]) -> None:
        """Remove multiple operators at once."""
        self._update_buckets()
        for op in ops:
            self._ops.remove(op)
        if not self._ops:
            self._active = False
        else:
            self._refresh_after_op_change()

    # =========================================================================
    # Optimizer Setup
    # =========================================================================

    def _normalize_threshold(self) -> Tuple[float, float]:
        """Convert threshold to (relative, absolute) tuple form."""
        if isinstance(self.yielder_threshold, tuple):
            return self.yielder_threshold
        return (float("inf"), self.yielder_threshold)

    def _reset_optimizer(self) -> None:
        """Reset optimizer with current operators (preserves learning schedules)."""
        if self.vid == 0 and not self.been_shortened:
            self.been_shortened = True

        method = self.optimizer_method.upper()
        threshold = self._normalize_threshold()

        if method == "SPSAP":
            self.optimizer = SPSAP(
                maxiter=self.step_size,
                cap_yielder=TreeVQA_Yielder(
                    window=self.cap_window_size,
                    warmup=self.cap_warmup,
                    operators=self.ops,
                    threshold=threshold,
                ),
                learning_rate=self.optimizer.eta,
                perturbation=self.optimizer.eps,
            )
        elif method == "COBYLAP":
            self.optimizer = COBYLAP(
                cap_yielder=SCIPY_YIELDER(
                    window=self.cap_window_size,
                    warmup=self.cap_warmup,
                    operators=self.ops,
                    threshold=threshold,
                    maxiter=self.step_size,
                    fun_pauli=self._get_eval_pauli_only(self.ansatz, self._target_op),
                ),
                optimize_config={"rhobeg": self.optimizer._config.rhobeg},
            )
        else:
            raise AlgorithmError(f"Unsupported optimizer: {self.optimizer_method}")

    def _calibrate_optimizer(self, optimizer) -> Optimizer:
        """Initialize optimizer from class or return existing instance."""
        if isinstance(optimizer, Optimizer):
            return optimizer

        method = self.optimizer_method.upper()
        threshold = self._normalize_threshold()

        if method == "SPSAP":
            return SPSAP(
                maxiter=self.step_size,
                cap_yielder=TreeVQA_Yielder(
                    window=self.cap_window_size,
                    warmup=self.cap_warmup,
                    operators=self.ops,
                    threshold=threshold,
                ),
                learning_rate=self.optimizer_configs.get("learning_rate"),
                perturbation=self.optimizer_configs.get("perturbation"),
            )
        elif method == "COBYLAP":
            return COBYLAP(
                cap_yielder=SCIPY_YIELDER(
                    window=self.cap_window_size,
                    warmup=self.cap_warmup,
                    operators=self.ops,
                    threshold=threshold,
                    maxiter=self.step_size,
                    fun_pauli=self._get_eval_pauli_only(self.ansatz, self._target_op),
                ),
                optimize_config=self.optimizer_configs,
            )

        raise AlgorithmError(f"Unsupported optimizer: {self.optimizer_method}")

    # =========================================================================
    # Simulation Setup
    # =========================================================================

    def _prepare_simulation(
        self, ansatz: QuantumCircuit
    ) -> Tuple[AerSimulator, QuantumCircuit, str, Any]:
        """Prepare simulator, ansatz, and optional Julia circuit.

        Returns:
            Tuple of (simulator, prepared_ansatz, method, julia_circuit).
        """
        method = "density_matrix" if self.noise_model else "statevector"
        simulator = AerSimulator(
            device=DEVICE, method=method, noise_model=self.noise_model
        )

        prepared_ansatz = ansatz.decompose()
        if self.noise_model:
            prepared_ansatz = transpile(
                prepared_ansatz,
                basis_gates=self.noise_model.basis_gates,
                optimization_level=3,
            )

        # Use Julia for large circuits
        julia_circuit = None
        if ansatz.num_qubits > JULIA_THRESHOLD:
            julia_circuit = create_efficientsu2_circuit(
                ansatz.num_qubits, 2, self.noise_model is not None
            )

        return simulator, prepared_ansatz, method, julia_circuit

    def _run_simulation(
        self,
        simulator: AerSimulator,
        ansatz: QuantumCircuit,
        parameters: np.ndarray,
        method: str,
    ):
        """Run simulation and return final quantum state."""
        bound_circuit = ansatz.assign_parameters(parameters, inplace=False)

        if method == "statevector":
            bound_circuit.save_statevector()
        else:
            bound_circuit.save_density_matrix()

        try:
            result = simulator.run(bound_circuit).result()
        except Exception as exc:
            raise AlgorithmError("Simulation failed!") from exc

        return result.data(0)[method]

    def _compute_pauli_values(
        self,
        parameters: np.ndarray,
        operator: OpTask,
        simulator: AerSimulator,
        ansatz: QuantumCircuit,
        method: str,
        julia_circuit,
    ) -> Dict[Pauli, float]:
        """Compute Pauli expectation values using simulator or Julia."""
        if julia_circuit:
            return estimate_pp_per_pauli(operator, julia_circuit, parameters)

        final_state = self._run_simulation(simulator, ansatz, parameters, method)
        operator_srt = operator.sort()
        pauli_terms = [Pauli(pauli) for pauli in operator_srt.paulis]
        vals = [final_state.expectation_value(pt) for pt in pauli_terms]
        return dict(zip(pauli_terms, vals))

    # =========================================================================
    # Energy Evaluation Functions
    # =========================================================================

    def _get_eval_per_pauli(
        self,
        ansatz: QuantumCircuit,
        operator: OpTask,
    ) -> Callable[[np.ndarray], Tuple[float, Dict[Pauli, float]]]:
        """Return function to evaluate energy and per-Pauli expectation values."""
        simulator, prepared_ansatz, method, julia_circuit = self._prepare_simulation(
            ansatz
        )
        operator_srt = operator.sort()

        def evaluate_energy_pauli(
            parameters: np.ndarray,
        ) -> Tuple[float, Dict[Pauli, float]]:
            pauli_values = self._compute_pauli_values(
                parameters, operator, simulator, prepared_ansatz, method, julia_circuit
            )
            for pt, val in pauli_values.items():
                self.op_history[pt].append(val)

            energy = np.dot(operator_srt.coeffs, list(pauli_values.values())).real
            return energy, pauli_values

        return evaluate_energy_pauli

    def _get_eval_pauli_only(
        self,
        ansatz: QuantumCircuit,
        operator: OpTask,
    ) -> Callable[[np.ndarray], Dict[Pauli, float]]:
        """Return function to evaluate only Pauli expectation values."""
        simulator, prepared_ansatz, method, julia_circuit = self._prepare_simulation(
            ansatz
        )

        def evaluate_pauli_only(parameters: np.ndarray) -> Dict[Pauli, float]:
            pauli_values = self._compute_pauli_values(
                parameters, operator, simulator, prepared_ansatz, method, julia_circuit
            )
            for pt, val in pauli_values.items():
                self.op_history[pt].append(val)
            return pauli_values

        return evaluate_pauli_only

    def _get_evaluate_energy(
        self,
        ansatz: QuantumCircuit,
        operator: OpTask,
    ) -> Callable[[np.ndarray], float]:
        """Return function to evaluate energy at given parameters.

        Supports parameter broadcasting for batch evaluation.
        """
        num_parameters = ansatz.num_parameters
        simulator, prepared_ansatz, method, julia_circuit = self._prepare_simulation(
            ansatz
        )
        operator_srt = operator.sort()

        def evaluate_energy(parameters: np.ndarray) -> np.ndarray | float:
            param_list = np.reshape(parameters, (-1, num_parameters)).tolist()

            values = []
            for params in param_list:
                if julia_circuit:
                    energy = estimate_pp(operator, julia_circuit, params)
                else:
                    final_state = self._run_simulation(
                        simulator, prepared_ansatz, params, method
                    )
                    pauli_vals = [
                        final_state.expectation_value(Pauli(p)).real
                        for p in operator_srt.paulis
                    ]
                    energy = np.dot(pauli_vals, operator_srt.coeffs).real
                values.append(energy)

            return values[0] if len(values) == 1 else values

        return evaluate_energy

    # =========================================================================
    # Optimization
    # =========================================================================

    def calibrate_spsa_hyperpara(self) -> SPSAHyperParams:
        """Calibrate SPSA hyperparameters for the current operator.

        Only valid when using SPSAP optimizer.

        Returns:
            SPSAHyperParams with calibrated learning rate and perturbation schedules.

        Raises:
            AlgorithmError: If not using SPSAP optimizer.
        """
        if self.optimizer_method.upper() != "SPSAP":
            raise AlgorithmError("SPSA calibration only available for SPSAP optimizer")

        return self.optimizer.get_hyperpara(
            self._get_evaluate_energy(self.ansatz, self._target_op), self.paras
        )

    def step_compute(
        self,
    ) -> Tuple[Any, Optional[List[OpTask]], Optional[List[OpTask]]]:
        """Run one batch of optimization steps.

        Returns:
            Tuple of (vqe_result, group1, group2). Groups are set if a split occurred.

        Raises:
            AssertionError: If called on inactive VQE.
        """
        assert self.is_active, "Cannot compute on inactive VQE"

        operator = self._target_op
        self._check_operator_ansatz(operator)

        start_time = time()
        was_updated = _set_default_batchsize(self.optimizer)

        optimizer_result, group1, group2 = self._run_optimizer_step(operator)

        if was_updated:
            self.optimizer.set_max_evals_grouped(None)

        elapsed = time() - start_time
        self.paras = optimizer_result.x
        self.culmulative_step += optimizer_result.nit
        self._k += optimizer_result.nit

        logging.info(
            "VQE%d: %d steps (total: %d), energy=%.6f",
            self.vid,
            optimizer_result.nit,
            self.culmulative_step,
            optimizer_result.fun,
        )

        if self.budget and self.culmulative_step >= self.budget:
            self.is_active = False
            logging.info(
                "VQE%d: Budget exhausted at %d steps", self.vid, self.culmulative_step
            )

        if self.optimizer_method.upper() == "SPSAP":
            self.optimizer.update_k_offset(self._k)

        return (
            self._build_vqe_result(self.ansatz, optimizer_result, None, elapsed),
            group1,
            group2,
        )

    def _run_optimizer_step(
        self, operator: OpTask
    ) -> Tuple[Any, Optional[List[OpTask]], Optional[List[OpTask]]]:
        """Execute optimizer minimize call based on method."""
        evaluate_energy = self._get_evaluate_energy(self.ansatz, operator)
        bounds = validate_bounds(self.ansatz)

        if self.optimizer_method.upper() == "SPSAP":
            evaluate_gradient = (
                self._get_evaluate_gradient(self.ansatz, operator)
                if self.gradient
                else None
            )
            return self.optimizer.minimize(
                fun=evaluate_energy,
                x0=self.paras,
                jac=evaluate_gradient,
                bounds=bounds,
                fun_pauli=self._get_eval_per_pauli(self.ansatz, operator),
            )
        else:
            return self.optimizer.minimize(fun=evaluate_energy, x0=self.paras)

    # =========================================================================
    # Result Generation
    # =========================================================================

    def check_appropriate(self, incoming_op: OpTask) -> bool:
        """Check if an incoming operator would be appropriate for this VQE.

        Uses slope of expectation value history to determine compatibility.

        Args:
            incoming_op: Operator to check for compatibility.

        Returns:
            True if operator's convergence slope is below threshold.
        """
        if self._k < self.cap_window_size:
            return False

        history = np.array(
            [
                self.op_history[Pauli(pauli)][-self.cap_window_size :]
                for pauli in incoming_op.paulis
            ]
        ).T

        energy_values = [np.dot(row, incoming_op.coeffs) for row in history]
        slope = np.polyfit(range(self.cap_window_size), energy_values, 1)[0]

        _, abs_threshold = self._normalize_threshold()
        return slope < abs_threshold / 2

    def get_result(self) -> VQEClusterResult:
        """Get complete result including Pauli expectation history.

        Returns:
            VQEClusterResult with full optimization history.
        """
        return VQEClusterResult(
            vid=self.vid,
            starting_step=self.starting_step,
            pauli_expectation=deepcopy(self.op_history),
            final_ops=deepcopy(self.ops),
            bucket_ids=deepcopy(self.bucket_ids),
            actual_step=self.culmulative_step,
        )
