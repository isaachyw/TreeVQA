"""
Main TreeVQA orchestration module.

This module provides the TreeVQA orchestrator
that manages parallel VQE optimization with automatic operator splitting.
"""

from math import pi
import logging
from itertools import count
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Union, List, Optional, Dict, Any, Tuple, Callable

import numpy as np
import scienceplots  # noqa: F401
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.primitives import Estimator
from qiskit_algorithms.utils import algorithm_globals
from qiskit_algorithms.optimizers.spsa import powerseries

from .optimizer.SPSAP import SPSAP, SPSAHyperParams
from .vqa.vqe_cluster import VQECluster
from .vqa.vqe_result import VQEClusterResult, SegmentEnergy
from .treevqa_helper import average_op_task
from .op_task import OpTask


@dataclass
class TreeVQAResult:
    """Results from a TreeVQA optimization run.

    Attributes:
        primary_result: List of checkpoint results, each mapping VQE ID to result.
        checkpoint_steps: Total steps at each checkpoint.
    """

    primary_result: List[Dict[int, VQEClusterResult]] = field(default_factory=list)
    checkpoint_steps: List[int] = field(default_factory=list)


@dataclass
class TreeVQAConfig:
    """Configuration for TreeVQA optimization.

    Attributes:
        seperate_ops: List of operators to optimize.
        cap_window_size: Window size for convergence monitoring.
        cap_warmup: Warmup iterations before convergence checks.
        cap_budget: Total step budget for optimization.
        ansatz: Parameterized quantum circuit.
        hf_bitstring: Hartree-Fock initial state (optional).
        seed: Random seed for reproducibility.
        log_file_name: Path to log file (optional).
        total_mode: If True, tracks total steps across all VQEs.
        vqe_step_size: Steps per optimization batch.
        check_interval: Checkpoint interval in total_mode.
        optimizer_method: "SPSAP" or "COBYLAP".
        optimizer_configs: Additional optimizer configuration.
        cap_yielder_threshold: Threshold for triggering splits.
        noisy: Enable noise simulation.
        gate_error: Gate error rate for noise model.
        cafqa_data: CAFQA grouping data for initialization (optional).
    """

    # Required parameters
    seperate_ops: List[OpTask]
    cap_window_size: int
    cap_warmup: int
    cap_budget: int
    ansatz: QuantumCircuit

    # Optional parameters
    hf_bitstring: Optional[np.ndarray] = None
    seed: int = 2002
    log_file_name: Optional[str] = None
    total_mode: bool = False
    vqe_step_size: int = 20
    check_interval: Optional[int] = None
    optimizer_method: str = "SPSAP"
    optimizer_configs: Optional[Dict[str, Any]] = None
    cap_yielder_threshold: float = -3e-6
    noisy: bool = False
    gate_error: float = 0.01
    cafqa_data: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.total_mode and self.check_interval is None:
            raise ValueError("check_interval required in total_mode")


class TreeVQA:
    """Main orchestrator for parallel VQE optimization with automatic splitting.

    TreeVQA monitors the convergence of multiple operators during VQE optimization
    and automatically splits them into separate groups when divergence is detected.

    Example:
        >>> config = TreeVQAConfig(
        ...     seperate_ops=operators,
        ...     cap_window_size=50,
        ...     cap_warmup=100,
        ...     cap_budget=1000,
        ...     ansatz=ansatz,
        ... )
        >>> treevqa = TreeVQA(config)
        >>> result = treevqa.average_vqe_sim()
    """

    def __init__(self, config: TreeVQAConfig) -> None:
        """Initialize TreeVQA orchestrator.

        Args:
            config: TreeVQA configuration parameters.
        """
        self.config = config
        self._init_from_config(config)

        self.spsa_hyperpara: Optional[SPSAHyperParams] = None
        self.vqe_map: Dict[int, VQECluster] = {}
        self.zombie_map: Dict[int, VQECluster] = {}
        self.counter = count(start=0, step=1)

        if self.hf_bitstring is not None and self.cafqa_data is None:
            self._apply_hartree_fock(self.ansatz, self.hf_bitstring)

    def _init_from_config(self, config: TreeVQAConfig) -> None:
        """Initialize attributes from config."""
        self.seperate_ops = config.seperate_ops
        self.cap_window_size = config.cap_window_size
        self.cap_warmup = config.cap_warmup
        self.shortened_warmup = max(config.cap_warmup // 3, config.cap_window_size)
        self.cap_budget = config.cap_budget
        self.estimator = Estimator
        self.hf_bitstring = config.hf_bitstring
        self.seed = config.seed
        self.ansatz = config.ansatz
        self.total_mode = config.total_mode
        self.vqe_step_size = config.vqe_step_size
        self.log_file_name = config.log_file_name
        self.optimizer_method = config.optimizer_method
        self.optimizer_configs = config.optimizer_configs or {}
        self.cap_yielder_threshold = config.cap_yielder_threshold
        self.noisy = config.noisy
        self.gate_error = config.gate_error
        self.cafqa_data = config.cafqa_data
        self.check_interval = config.check_interval if self.total_mode else None

    @staticmethod
    def _apply_hartree_fock(
        ansatz: QuantumCircuit, hf_bitstring: Union[List[bool], np.ndarray]
    ) -> None:
        """Prepend Hartree-Fock state preparation to ansatz.

        Args:
            ansatz: Quantum circuit to modify in-place.
            hf_bitstring: Hartree-Fock occupation bitstring.
        """
        hf_circuit = QuantumCircuit(ansatz.num_qubits)
        for i, bit in enumerate(hf_bitstring):
            if bit:
                hf_circuit.x(i)
        ansatz.compose(hf_circuit, front=True, inplace=True)

    def _create_vqe_cluster(
        self,
        ops: List[OpTask],
        initial_point: Optional[np.ndarray],
        parrent_id: int,
        starting_step: int,
        culmulative_step: int,
        warmup: Optional[int] = None,
        optimizer_configs: Optional[Dict[str, Any]] = None,
    ) -> VQECluster:
        """Factory method to create VQECluster with common parameters.

        Args:
            ops: Operators for this VQE instance.
            initial_point: Initial parameters (None for random).
            parrent_id: Parent VQE ID (-1 for root).
            starting_step: Step count when this VQE was created.
            culmulative_step: Total steps from root.
            warmup: Override warmup period (None to use default).
            optimizer_configs: Optimizer configuration (None to use default).

        Returns:
            Configured VQECluster instance.
        """
        return VQECluster(
            estimator=self.estimator(),
            ansatz=self.ansatz,
            optimizer=SPSAP,
            averager=average_op_task,
            budget=None if self.total_mode else self.cap_budget,
            ops=ops,
            vid=next(self.counter),
            culmulative_step=culmulative_step,
            starting_step=starting_step,
            initial_point=initial_point,
            cap_window_size=self.cap_window_size,
            cap_warmup=warmup or self.cap_warmup,
            parrent_id=parrent_id,
            step_size=self.vqe_step_size,
            optimizer_method=self.optimizer_method,
            optimizer_configs=optimizer_configs or self.optimizer_configs,
            yielder_threshold=self.cap_yielder_threshold,
            noisy=self.noisy,
            gate_error=self.gate_error,
        )

    def _needs_spsa_calibration(self) -> bool:
        """Check if SPSA hyperparameters need calibration."""
        return (
            self.optimizer_method.upper() == "SPSAP"
            and not self.optimizer_configs.get("learning_rate")
            and not self.optimizer_configs.get("perturbation")
        )

    def _calibrate_and_update_vqes(self) -> None:
        """Calibrate SPSA and update all VQE instances."""
        if not self._needs_spsa_calibration():
            return

        first_vqe = next(iter(self.vqe_map.values()))
        self.spsa_hyperpara = first_vqe.calibrate_spsa_hyperpara()

        for vqe_cluster in self.vqe_map.values():
            lr_func, pert_func = self._get_learning_rate_funcs(self.spsa_hyperpara, 0)
            vqe_cluster.optimizer_configs.update(
                {
                    "learning_rate": lr_func(),
                    "perturbation": pert_func(),
                }
            )
            vqe_cluster.optimizer = vqe_cluster._calibrate_optimizer(SPSAP)

    def boot_treevqa(self) -> None:
        """Initialize VQE instances for optimization."""
        if self.cafqa_data is not None:
            self._boot_with_cafqa()
        else:
            self._boot_standard()

        self._calibrate_and_update_vqes()

    def _boot_with_cafqa(self) -> None:
        """Initialize VQEs from CAFQA grouping data."""
        groups = self.cafqa_data["groups"]
        ks_bests = self.cafqa_data["group_ks_bests"]

        logging.info("Creating %d VQE instances from CAFQA groups", len(groups))

        for group_idx, op_task_indices in enumerate(groups):
            group_ops = [self.seperate_ops[i] for i in op_task_indices]
            init_point = [i * pi / 2 for i in ks_bests[group_idx]]

            vqe = self._create_vqe_cluster(
                group_ops,
                init_point,
                parrent_id=-1,
                starting_step=0,
                culmulative_step=0,
            )
            self.vqe_map[vqe.vid] = vqe
            logging.info(
                "Created VQE %d for group %d (%d ops)",
                vqe.vid,
                group_idx,
                len(group_ops),
            )

    def _boot_standard(self) -> None:
        """Initialize single VQE with all operators."""
        vqe = self._create_vqe_cluster(
            self.seperate_ops, None, parrent_id=-1, starting_step=0, culmulative_step=0
        )
        self.vqe_map[vqe.vid] = vqe
        logging.info("Created initial VQE with %d operators", len(self.seperate_ops))

    def average_vqe_sim(self) -> TreeVQAResult:
        """Run parallel VQE optimization.

        Returns:
            TreeVQAResult containing optimization history and final results.
        """
        algorithm_globals.random_seed = self.seed
        treevqa_result = TreeVQAResult()
        check_counter = count(start=0, step=1)

        self.boot_treevqa()

        while self.vqe_map:
            new_vqes = self._run_optimization_step()
            self.vqe_map.update(new_vqes)
            self._retire_inactive_vqes()

            if self.total_mode:
                total_step = sum(v.individual_step for v in self.vqe_map.values())
                if total_step >= self.cap_budget:
                    self._finalize_remaining_vqes()
                    break

                logging.info("Total steps: %d", total_step)
                if self._should_checkpoint(check_counter, new_vqes):
                    treevqa_result.primary_result.append(
                        {
                            v.vid: v.get_result()
                            for v in self.vqe_map.values()
                            if v.is_active
                        }
                    )
                    treevqa_result.checkpoint_steps.append(total_step)

        logging.info("Optimization complete: %d VQE buckets", len(self.zombie_map))
        treevqa_result.primary_result.append(
            {v.vid: v.get_result() for v in self.zombie_map.values()}
        )
        if self.total_mode:
            treevqa_result.checkpoint_steps.append(
                sum(v.individual_step for v in self.zombie_map.values())
            )
        return treevqa_result

    def _run_optimization_step(self) -> Dict[int, VQECluster]:
        """Run one step for all active VQEs, handle splits.

        Returns:
            Dictionary of newly created VQEs from splits.
        """
        new_vqes: Dict[int, VQECluster] = {}

        for vid in list(self.vqe_map.keys())[::-1]:
            vqe = self.vqe_map[vid]
            logging.info("VQE %d computing", vqe.vid)

            result, group1, group2 = vqe.step_compute()
            if group1 is not None and group2 is not None:
                vqe.remove_multiple_ops(group2)
                new_vqe = self._create_split_vqe(vqe, group2, result.optimal_point)
                new_vqes[new_vqe.vid] = new_vqe

        return new_vqes

    def _create_split_vqe(
        self, parent: VQECluster, ops: List[OpTask], initial_point: np.ndarray
    ) -> VQECluster:
        """Create new VQE from split with updated learning rates.

        Args:
            parent: Parent VQE that triggered the split.
            ops: Operators for the new VQE.
            initial_point: Initial parameters (inherited from parent).

        Returns:
            New VQECluster instance.
        """
        new_configs = self.optimizer_configs.copy()

        if self.optimizer_method.upper() == "SPSAP" and self.spsa_hyperpara:
            lr_func, pert_func = self._get_learning_rate_funcs(
                self.spsa_hyperpara, parent.culmulative_step + 1
            )
            new_configs.update(
                {
                    "learning_rate": lr_func(),
                    "perturbation": pert_func(),
                }
            )
        return self._create_vqe_cluster(
            ops,
            initial_point,
            parrent_id=parent.vid,
            starting_step=parent.culmulative_step,
            culmulative_step=parent.culmulative_step,
            warmup=self.shortened_warmup,
            optimizer_configs=new_configs,
        )

    def _retire_inactive_vqes(self) -> None:
        """Move inactive VQEs to zombie_map."""
        for vid in list(self.vqe_map.keys()):
            if not self.vqe_map[vid].is_active:
                self.zombie_map[vid] = self.vqe_map.pop(vid)

    def _finalize_remaining_vqes(self) -> None:
        """Run final step for new VQEs and retire all."""
        logging.info("Finalizing remaining VQEs")
        for vqe in self.vqe_map.values():
            if len(vqe.op_history) == 0:
                vqe.step_compute()
            vqe.is_active = False
            self.zombie_map[vqe.vid] = vqe
        self.vqe_map.clear()

    def _should_checkpoint(self, counter: count, new_vqes: Dict) -> bool:
        """Check if checkpoint should be recorded."""
        return (
            self.check_interval is not None
            and next(counter) % self.check_interval == 0
            and len(new_vqes) == 0
        )

    def _get_learning_rate_funcs(
        self, hyper_para: Optional[SPSAHyperParams], starting_point: int
    ) -> Tuple[Callable[[], float], Callable[[], float]]:
        """Get learning rate and perturbation generator functions.

        Args:
            hyper_para: SPSA hyperparameters (None to use config values).
            starting_point: Iteration offset for schedule.

        Returns:
            Tuple of (learning_rate_factory, perturbation_factory) functions.
        """
        if hyper_para is None:
            return (
                lambda: self.optimizer_configs.get("learning_rate"),
                lambda: self.optimizer_configs.get("perturbation"),
            )

        def lr_func():
            return powerseries(
                hyper_para.a,
                hyper_para.alpha,
                hyper_para.stability_constant + starting_point,
            )

        def pert_func():
            return powerseries(hyper_para.c, hyper_para.gamma, starting_point)

        return lr_func, pert_func


def treevqa_worker(treevqa: TreeVQA, op_task: OpTask, return_dict: Dict, lock) -> None:
    """Worker function for multiprocessing.

    Args:
        treevqa: TreeVQA orchestrator instance.
        op_task: Operator for identification.
        return_dict: Shared dictionary for results.
        lock: Multiprocessing lock.
    """
    result = treevqa.average_vqe_sim()
    with lock:
        return_dict[op_task.op_id] = result


def plot_overview(
    energy_by_vqe: List[Dict[int, List[SegmentEnergy]]], max_iter: int, fig_dir: str
) -> None:
    """Plot combined energy trajectories across all VQEs.

    Args:
        energy_by_vqe: Energy segments by operator ID for each VQE.
        max_iter: Maximum iteration count for x-axis.
        fig_dir: Directory to save the figure.
    """
    energy_by_op: Dict[int, List[float]] = defaultdict(list)
    op_ids = sorted({op_id for vqe in energy_by_vqe for op_id in vqe.keys()})

    for op_id in op_ids:
        energy_by_op[op_id] = [0.0] * max_iter

    for vqe in energy_by_vqe:
        for op_id, segments in vqe.items():
            for seg in segments:
                energy_by_op[op_id][seg.start : seg.end] = seg.energy

    colormap = plt.cm.get_cmap("tab10")
    color_map = {op_id: colormap(i % colormap.N) for i, op_id in enumerate(op_ids)}

    with plt.style.context(("science", "grid")):
        for op_id in op_ids:
            energy = np.array(energy_by_op[op_id])
            plt.plot(
                np.arange(max_iter),
                np.where(energy == 0, np.nan, energy),
                label=f"{op_id}",
                color=color_map[op_id],
            )
        plt.xlabel("Iteration")
        plt.ylabel("Energy")
        plt.title("VQE Overview")
        plt.legend(prop={"size": 6, "weight": "bold"})
        plt.savefig(f"{fig_dir}/overview.png", dpi=250)
        plt.close()
