"for profiling single VQE"

import os
import click
import logging
from typing import List
import numpy as np
from TreeVQA.application.molecule_application import MoleculeExperiment
from TreeVQA.application.ising_application import IsingModelExperiment
from TreeVQA.application.maxcut_application import MaxCutExperiment
from TreeVQA.application.power_application import PowerExperiment
from TreeVQA.TreeVQA import TreeVQAResult
from TreeVQA.vqa.vqe_result import VQEClusterResult
from TreeVQA.application.util import save_vqe_intermediate_data

logging.getLogger("qiskit").setLevel(logging.ERROR)
COMMON_OPTIONS = [
    click.option("-v", "--vqe_iter", default=400, type=int, help="VQE iterations"),
    click.option("-s", "--seed", default=2002, type=int, help="Random seed"),
    click.option("-w", "--cap_window", default=30, type=int, help="CAP window size"),
    click.option("-u", "--cap_warmup", default=200, type=int, help="CAP warmup period"),
    click.option("-r", "--repetition", default=1, type=int, help="Ansatz repetitions"),
    click.option("-c", "--cafqa", is_flag=True, help="Enable CAFQA initialization"),
    click.option("-cb", "--cafqabudget", default=30, type=int, help="CAFQA budget"),
    click.option("--total_mode", is_flag=True, help="Enable total shot mode"),
    click.option("--vqe_step_size", default=20, type=int, help="VQE step size"),
    click.option(
        "--optimizer_config",
        multiple=True,
        help="Optimizer configuration in format key=value (e.g., learning_rate=0.1)",
    ),
    click.option(
        "--optimizer_method",
        type=click.Choice(["SPSAP", "COBYLAP"], case_sensitive=False),
        default="SPSAP",
        help="Optimizer method to use",
    ),
    click.option(
        "--threshold",
        type=(float, click.FloatRange(max=0.0)),
        help="CAP yielder threshold(relative,absolute)",
    ),
    click.option(
        "--check_interval",
        default=5,
        type=int,
        help="Loss logging interval in major loop interval",
    ),
    click.option(
        "--ansatz",
        type=click.Choice(["HEA", "UCCSD", "QAOA"], case_sensitive=False),
        default="HEA",
        help="Select the ansatz type: HEA, UCCSD, or QAOA.",
    ),
    click.option(
        "--result_dir",
        default="results",
        type=str,
        help="Directory to save the results",
    ),
    click.option(
        "--noisy",
        is_flag=True,
    ),
    click.option(
        "--gate_error",
        type=float,
        default=0.01,
        help="Gate error rate for the noise model",
    ),
]

MOLECULE_OPTIONS = [
    click.option("-m", "--molecule_name", default="H2", help="Molecule name"),
    click.option("--molecule_slice", default="0:2,-2:", help="Bond length slices"),
    click.option("--freeze", is_flag=True, help="Freeze core electrons"),
    click.option(
        "--truncate_ratio",
        default=0.005,
        type=float,
        help="Truncate ratio of pauli terms",
    ),
]

ISING_OPTIONS = [
    click.option("-n", "--num_node", default=2, type=int, help="Number of nodes"),
    click.option(
        "--magnetic_fields",
        default="0.0,0.5,1.0",
        help="Comma-separated magnetic field values",
    ),
    click.option(
        "--model_type",
        type=click.Choice(["heisenberg", "ising"], case_sensitive=False),
        default="ising",
        help="Model type: heisenberg or ising",
    ),
]

MAXCUT_OPTIONS = [
    click.option(
        "--num_vertices",
        default=10,
        type=int,
        help="Number of vertices in the graph",
    ),
    click.option(
        "--graph_slice",
        default="0:100:1",
        help="Graph indices to analyze (format: start:end:step)",
    ),
    click.option(
        "--weight_range",
        default="0.5:1.5:0.05",
        help="Edge weight range and step size (format: min:max:step)",
    ),
]


def add_options(options):
    def decorator(func):
        for option in reversed(options):
            func = option(func)
        return func

    return decorator


@click.command()
@click.option(
    "-p",
    "--problem",
    type=click.Choice(["molecule", "ising", "maxcut", "power"], case_sensitive=False),
    required=True,
    help="Problem type to solve",
)
@add_options(COMMON_OPTIONS + MOLECULE_OPTIONS + ISING_OPTIONS + MAXCUT_OPTIONS)
def main(**kwargs):
    """Quantum Experiment CLI"""
    print("run single vqe")
    problem_type = kwargs.pop("problem")

    # Process optimizer configurations
    optimizer_configs = {}
    if kwargs.get("optimizer_config"):
        for config_str in kwargs.pop("optimizer_config"):
            if "=" in config_str:
                key, value = config_str.split("=", 1)
                # Try to convert to appropriate type
                try:
                    # Try float first
                    optimizer_configs[key.strip()] = float(value.strip())
                except ValueError:
                    # Keep as string if conversion fails
                    optimizer_configs[key.strip()] = value.strip()

    params = {k: v for k, v in kwargs.items() if v is not None}
    params["optimizer_configs"] = optimizer_configs if optimizer_configs else None

    # Handle ansatz compatibility
    if problem_type == "ising" and params["ansatz"] == "UCCSD":
        raise ValueError("UCCSD ansatz is not supported for Ising model.")
    if problem_type == "maxcut" and params["ansatz"] != "QAOA":
        raise ValueError("Only QAOA ansatz is supported for MaxCut problems.")

    # Instantiate appropriate experiment
    experiment_map = {
        "molecule": MoleculeExperiment,
        "ising": IsingModelExperiment,
        "maxcut": MaxCutExperiment,
        "power": PowerExperiment,
    }
    experiment = experiment_map[problem_type](params)

    # Run experiment workflow
    experiment.load_data()
    if params.get("cafqa", False):
        cafqa_data = experiment.run_cafqa_initialization(split=True)
    else:
        cafqa_data = None
    separate_results: List[TreeVQAResult] = experiment.run_parallel_vqe(cafqa_data)
    for single_vqe_result, op_task in zip(separate_results, experiment.op_tasks):
        checkpoint_indices = single_vqe_result.checkpoint_steps
        intermediate_data = []
        for checkpoint_data in single_vqe_result.primary_result:
            # there should only be one vqe bucket id with 0
            vqe0_result = checkpoint_data[0].final_energy
            intermediate_data.append(vqe0_result)
        intermediate_data = np.array(intermediate_data)
        # figure out the parent directory of result_dir_base
        result_dir_base = os.path.dirname(experiment.result_dir_base)
        save_vqe_intermediate_data(
            intermediate_data=intermediate_data,
            checkpoint_indices=checkpoint_indices,
            result_dir=result_dir_base,
            op_id=op_task.op_id,
        )


if __name__ == "__main__":
    main()
