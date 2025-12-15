# File: treevqa_application.py
"""Base class for TreeVQA-based quantum experiments."""

from typing import Any, Dict, List, Optional, Tuple
from threading import Thread

import abc
import logging
import os
import json
import numpy as np
import scienceplots
import matplotlib.pyplot as plt
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import Statevector, StabilizerState
from TreeVQA.TreeVQA import TreeVQA, TreeVQAResult, plot_overview, TreeVQAConfig
from TreeVQA.vqa.vqe_result import VQEClusterResult
from TreeVQA.op_task import OpTask
from .util import visualize_energy_convergence
from ..clapton.clapton.clapton import claptonize
from ..clapton.clapton.ansatzes import circular_ansatz


class TreeVQAApplication(abc.ABC):
    """Base class for TreeVQA-based quantum experiments."""

    DEFAULT_PARAMS = {
        "vqe_iter": 400,
        "seed": 1925,  # in memory of Schrödinger equation
        "cap_window": 30,
        "cap_warmup": 200,
        "optimizer_method": "SPSAP",
        "optimizer_configs": None,
        "repetition": 1,
        "cafqa": False,
        "cafqabudget": 40,
        "total_mode": False,
        "vqe_step_size": 20,
        "threshold": -3e-6,
        "check_interval": 5,
        "noisy": False,
    }

    def __init__(self, params: Dict):
        self.params = {**self.DEFAULT_PARAMS, **params}
        self._validate_params()
        self._setup_infrastructure()
        self.op_tasks: List[OpTask] = []
        self.ref_values: List[float] = []
        self.hf_bits: Optional[List[bool]] = None
        self.ansatz: EfficientSU2 = None
        self.avg_hinfo = None
        self.fig_params: Dict[str, str] = {
            "x_label": "Operator ID",
            "title": "Energy Comparison",
        }

    @property
    @abc.abstractmethod
    def result_dir_base(self) -> str:
        """Base directory for results (implemented by subclasses)"""

    @abc.abstractmethod
    def load_data(self) -> None:
        """Load application-specific data (implemented by subclasses)"""

    def _validate_params(self) -> None:
        """Validate common parameters"""
        if self.params["vqe_iter"] <= 0:
            raise ValueError("VQE iterations must be positive")
        if self.params["cap_window"] > self.params["cap_warmup"]:
            raise ValueError("Warmup period must be greater than window size")

    def _setup_infrastructure(self) -> None:
        """Initialize common logging infrastructure"""
        os.makedirs(self.result_dir, exist_ok=True)
        self.log_file = f"{self.result_dir}/experiment.log"

        if os.path.exists(self.log_file):
            os.remove(self.log_file)

        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format="%(message)s",
            filemode="w",
            force=True,
        )
        logging.info("Experiment parameters: %s", self.params)
        logging.getLogger("qiskit").setLevel(logging.ERROR)
        logging.getLogger("qiskit.transpiler").setLevel(logging.ERROR)
        logging.getLogger("qiskit.compiler").setLevel(logging.ERROR)
        print(f"Logging to {self.log_file}")

    @property
    def result_dir(self) -> str:
        """Generate result directory path"""
        return (
            f"{self.result_dir_base}/"
            f"VQE{self.params['vqe_iter']}_"
            f"W{self.params['cap_window']}_"
            f"WA{self.params['cap_warmup']}_"
            f"T{self.params['threshold']}_"
            f"{self.params['ansatz']}"
        )

    def run_cafqa_initialization(self, split=False) -> Optional[Dict[str, Any]]:
        """Initialize parameters using CAFQA if enabled and group by equivalent states"""
        benchmark_dir = os.path.dirname(self.result_dir_base)
        if not self.params["cafqa"]:
            return None

        init_points, cafqa_energies, cafqa_states, ks_bests = [], [], [], []

        groups: List[List[int]] = []
        group_ks_bests = []
        json_file_name = f"cafqa_data_{self.params['repetition']}.json"
        if not os.path.exists(os.path.join(benchmark_dir, json_file_name)):
            with open(os.path.join(benchmark_dir, json_file_name), "w") as f:
                json.dump({}, f)
        for op_task in self.op_tasks:
            num_qubits = op_task.num_qubits
            with open(
                os.path.join(benchmark_dir, json_file_name), "r", encoding="utf-8"
            ) as f:
                data = json.load(f)
            if str(op_task.op_id) in data:
                ks_bests.append(data[str(op_task.op_id)]["ks_best"])
                init_points.append(
                    [k * np.pi / 2 for k in data[str(op_task.op_id)]["ks_best"]]
                )
                cafqa_energies.append(data[str(op_task.op_id)]["energy_best"])
                group_ks_bests.append(data[str(op_task.op_id)]["ks_best"])

            else:
                vqe_circ = circular_ansatz(
                    N=num_qubits, reps=self.params["repetition"], fix_2q=True
                )
                ks_best, _, energy_best = claptonize(
                    [p_str[::-1].to_label() for p_str in op_task.paulis],
                    op_task.coeffs.real,
                    vqe_circ,
                    n_proc=4,
                    budget=self.params["cafqabudget"],
                    n_starts=4,
                )

                ks_bests.append(ks_best)
                init_points.append([k * np.pi / 2 for k in ks_best])
                cafqa_energies.append(energy_best.item())
                with open(
                    os.path.join(benchmark_dir, json_file_name), "r", encoding="utf-8"
                ) as f:
                    data = json.load(f)
                    data[str(op_task.op_id)] = {
                        "ks_best": ks_best,
                        "energy_best": energy_best.item(),
                    }
                with open(
                    os.path.join(benchmark_dir, json_file_name), "w", encoding="utf-8"
                ) as f:
                    json.dump(data, f, indent=4)

            hwe = EfficientSU2(
                num_qubits=num_qubits,
                reps=self.params["repetition"],
                entanglement="circular",
            )
            hwe.assign_parameters(init_points[-1], inplace=True)
            q_stab = StabilizerState(hwe)
            cafqa_states.append(q_stab)

        if split:
            return {
                "init_points": init_points,
                "cafqa_energies": cafqa_energies,
                "group_ks_bests": group_ks_bests,
            }

        for i, state in enumerate(cafqa_states):
            found_group = False
            for group in groups:
                representative_idx = group[0]
                if state.equiv(cafqa_states[representative_idx]):
                    group.append(i)
                    found_group = True
                    break

            if not found_group:
                groups.append([i])
                group_ks_bests.append(ks_bests[i])

        logging.info(
            "CAFQA initialization complete with %d equivalent groups", len(groups)
        )
        return {
            "groups": groups,
            "group_ks_bests": group_ks_bests,
            "init_points": init_points,
            "cafqa_energies": cafqa_energies,
        }

    def execute_treevqa(
        self, cafqa_data: Optional[Dict[str, Any]] = None
    ) -> TreeVQAResult:
        """Execute the main TreeVQA experiment"""
        treevqa_config = TreeVQAConfig(
            seperate_ops=self.op_tasks.copy(),
            cap_window_size=self.params["cap_window"],
            cap_warmup=self.params["cap_warmup"],
            cap_budget=self.params["vqe_iter"],
            ansatz=self.ansatz,
            hf_bitstring=np.array(self.hf_bits) if self.hf_bits is not None else None,
            seed=self.params["seed"],
            log_file_name=f"{self.result_dir}/treevqa_data.npy",
            total_mode=self.params["total_mode"],
            vqe_step_size=self.params["vqe_step_size"],
            check_interval=self.params["check_interval"],
            optimizer_method=self.params["optimizer_method"],
            optimizer_configs=self.params["optimizer_configs"],
            cap_yielder_threshold=self.params["threshold"],
            noisy=self.params.get("noisy", False),
            gate_error=self.params.get("gate_error", 0.0),
            cafqa_data=cafqa_data,
        )
        treevqa = TreeVQA(config=treevqa_config)
        return treevqa.average_vqe_sim()

    def run_parallel_vqe(self, cafqa_data: Dict = None) -> List[TreeVQAResult]:
        """Run parallel VQE experiments using threads"""
        results = []

        def thread_task(index: int, op_task: OpTask, init_point: List[float]):
            result = self._run_single_vqe(op_task, self.params["vqe_iter"], init_point)
            if result is not None:
                results.append(result)

        if cafqa_data is not None:
            for i, op_task in enumerate(self.op_tasks):
                init_point = cafqa_data["init_points"][i]
                group_ks_best = cafqa_data["group_ks_bests"][i]
                cafqa_energies = cafqa_data["cafqa_energies"][i]
                cafqa_data_modified = {
                    "init_points": [init_point],
                    "groups": [[0]],
                    "group_ks_bests": [group_ks_best],
                    "cafqa_energies": [cafqa_energies],
                }
                thread_task(i, op_task, cafqa_data_modified)
        else:
            for i, op_task in enumerate(self.op_tasks):
                thread_task(i, op_task, None)

        return results

    def _run_single_vqe(
        self, op_task: OpTask, budget: int, cafqa_data: Dict = None
    ) -> TreeVQAResult:
        """Run a single VQE instance"""
        treevqa_config = TreeVQAConfig(
            seperate_ops=[op_task],
            cap_window_size=self.params["cap_window"],
            cap_warmup=self.params["cap_warmup"],
            cap_budget=budget,
            ansatz=self.ansatz,
            seed=self.params["seed"],
            log_file_name=None,
            hf_bitstring=np.array(self.hf_bits) if self.hf_bits is not None else None,
            total_mode=self.params["total_mode"],
            vqe_step_size=self.params["vqe_step_size"],
            check_interval=self.params["check_interval"],
            optimizer_method=self.params["optimizer_method"],
            optimizer_configs=self.params["optimizer_configs"],
            cap_yielder_threshold=self.params["threshold"],
            noisy=self.params.get("noisy", False),
            gate_error=self.params.get("gate_error", 0.0),
            cafqa_data=cafqa_data,
        )

        treevqa = TreeVQA(config=treevqa_config)
        result = treevqa.average_vqe_sim()

        return result

    def visualize_results(self, treevqa_result: TreeVQAResult) -> None:
        """Visualize and save experiment results"""
        optimized_energies = self._optimize_treevqa_results(
            treevqa_result.primary_result[-1], self.op_tasks
        )
        optimized_energies_list = [optimized_energies[op.op_id] for op in self.op_tasks]
        final_op_ids, final_energies, intermediate_result = (
            self._process_treevqa_results(treevqa_result, optimized_energies_list)
        )

        self._plot_energy_comparison(
            final_op_ids, final_energies, optimized_energies_list
        )

    def _process_treevqa_results(
        self, treevqa_result: TreeVQAResult, optimized_energies: List[float]
    ) -> Tuple[List[float], List[float], np.ndarray]:
        """Process TreeVQA results into plottable format"""
        check_points = treevqa_result.checkpoint_steps
        intermediate_result, final_treevqa_result = (
            treevqa_result.primary_result[:],
            treevqa_result.primary_result[-1],
        )
        intermediate_data, optimized_intermediate_data = [], []
        for checkpoint_data in intermediate_result:
            final_op_ids: List[float] = []
            final_energies: List[float] = []
            for vid, result in checkpoint_data.items():
                final_op_ids.extend(float(op.op_id) for op in result.final_ops)
                final_energies.extend(result.final_energy)

                sorted_pairs = sorted(
                    zip(final_op_ids, final_energies), key=lambda x: x[0]
                )
                final_op_ids = [x[0] for x in sorted_pairs]
                final_energies = [x[1] for x in sorted_pairs]
            optimized_intermediate_energies = self._optimize_treevqa_results(
                checkpoint_data, self.op_tasks
            )
            optimized_intermediate_data.append(
                list(optimized_intermediate_energies.values())
            )
            intermediate_data.append(final_energies)
        intermediate_data_np = np.array(intermediate_data)
        np.save(
            os.path.join(self.result_dir, "op_ids.npy"),
            np.array([float(op.op_id) for op in self.op_tasks]),
        )
        visualize_energy_convergence(
            intermediate_data_np,
            np.array(self.ref_values),
            self.result_dir,
            np.array(check_points),
            optimized_energies,
            optimized_intermediate_data,
        )
        final_op_ids, final_energies = [], []
        energy_data = []
        max_iter = 0

        for vid, result in final_treevqa_result.items():
            final_op_ids.extend([float(op.op_id) for op in result.final_ops])
            final_energies.extend(vqe_energys := result.final_energy)
            energy_data.append(result.plot_energys(self.result_dir))
            max_iter = max(max_iter, result.actual_step)
            logging.info("%d: final energy: %s", vid, vqe_energys)

        plot_overview(energy_data, max_iter, self.result_dir)
        return (
            sorted(final_op_ids),
            [e for _, e in sorted(zip(final_op_ids, final_energies))],
            intermediate_data_np,
        )

    def _optimize_treevqa_results(
        self, treevqa_result: Dict[int, VQEClusterResult], op_tasks: List[OpTask]
    ) -> Dict[float, float]:
        """Calculate optimized energies for all operators"""
        pauli_terms = op_tasks[0].paulis
        final_exp_values = [
            [vqe.pauli_expectation[p][-1] for p in pauli_terms]
            for vqe in treevqa_result.values()
            if vqe.pauli_expectation
        ]

        if not final_exp_values:
            return {float(op.op_id): np.inf for op in op_tasks}

        final_exp_matrix = np.array(final_exp_values, dtype=np.complex128)
        coeff_matrix = np.array([op.coeffs for op in op_tasks], dtype=np.complex128).T
        energies_all = final_exp_matrix @ coeff_matrix
        min_energies = np.min(energies_all.real, axis=0)

        return {float(op.op_id): energy for op, energy in zip(op_tasks, min_energies)}

    def _plot_energy_comparison(
        self,
        op_ids: List[float],
        treevqa_energies: List[float],
        optimized_energies: List[float],
    ) -> None:
        """Generate energy comparison plot"""
        plt.figure(figsize=(10, 6))

        plt.plot(op_ids, self.ref_values, "o-", label="Reference")
        plt.plot(op_ids, treevqa_energies, "s--", label="TreeVQA Result")
        plt.plot(op_ids, optimized_energies, "v-.", label="Optimized TreeVQA")

        plt.xlabel(self.fig_params["x_label"])
        plt.ylabel("Energy (Hartree)")
        plt.title(self.fig_params["title"])
        plt.legend()
        plt.grid(True)
        plt.savefig(
            f"{self.result_dir}/energy_comparison.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
