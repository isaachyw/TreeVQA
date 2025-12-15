import numpy as np
import argparse
from typing import List, Tuple
import os
import json
import time

# Scientific plotting setup
import matplotlib.pyplot as plt
import scienceplots

plt.style.use("science")

# Import required functions
from TreeVQA.treevqa_helper import (
    solve_groundstate_numpy,
    molecule_to_op,
    get_molecule_coords,
)
from TreeVQA.application.Ising_model import (
    load_sparseOp_with_id,
    load_sparseOp_with_id_square,
    load_sparseOp_heisenberg_id,
)
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.units import DistanceUnit


def plot_molecule_energy(
    molecule_names: List[str],
    starts: List[float] = 0.2,
    ends: List[float] = 2.5,
    granularities: List[float] = 0.01,
    plot: bool = True,
):
    """Plot ground state energy for molecules."""
    for molecule_name, start, end, granularity in zip(
        molecule_names, starts, ends, granularities
    ):
        # Setup bond lengths
        print(
            f"solving ground state of {molecule_name} from {start} to {end} with granularity {granularity}"
        )
        num_points = int((end - start) / granularity) + 2
        bond_lengths = np.round(np.linspace(start, end, num_points), decimals=3)
        print("loading molecule coordinates and Hamiltonians")
        # Get molecule coordinates and Hamiltonians
        molecule_coords = [
            get_molecule_coords(molecule_name, bond) for bond in bond_lengths
        ]
        Hinfos = [
            molecule_to_op(coord, DistanceUnit.ANGSTROM, freeze_core=True)
            for coord in molecule_coords
        ]
        pauli_ops = [
            SparsePauliOp(data=hinfo[1], coeffs=hinfo[0]).sort() for hinfo in Hinfos
        ]
        # Calculate ground energies
        ground_energy = []
        molecule_ground_state = []
        total_tasks = len(pauli_ops)
        print("solving ground state of each Hamiltonian")
        for i, pauli_op in enumerate(pauli_ops):
            print(f"Task {i + 1}/{total_tasks}")
            start_time = time.time()
            result = solve_groundstate_numpy(pauli_op)
            molecule_energy = result.eigenvalue.real
            molecule_ground_state.append(result.eigenstate)
            ground_energy.append(
                float(molecule_energy)
            )  # Convert to float for JSON serialization
            print(f"Time taken: {time.time() - start_time:.2f}s")

        # Save results with consistent float formatting
        energy_bondlength_map = {
            f"{round(float(bond), 2):.2f}": float(
                energy
            )  # Format bond length to 3 decimal places
            for bond, energy in zip(bond_lengths, ground_energy)
        }
        with open(f"{molecule_name}.json", "w") as f:
            json.dump(energy_bondlength_map, f, indent=4)
        if plot:
            with plt.style.context(["science", "light"]):
                plt.figure(figsize=(10, 8))
                plt.plot(bond_lengths, ground_energy)
                plt.xlabel("Bond Length (Å)")
                plt.ylabel("Ground State Energy (Hartree)")
                plt.xticks(np.linspace(min(bond_lengths), max(bond_lengths), 9))
                plt.title(f"{molecule_name} energy vs bond length")
                plt.savefig(
                    f"ground-state/{molecule_name}.png", dpi=300, bbox_inches="tight"
                )
                plt.close()


def plot_physical_energy(
    sizes: List[int] = [3],
    starts: List[float] = [0.50],
    ends: List[float] = [1.50],
    granularities: List[float] = [0.01],
    plot: bool = True,
    model_type: List[str] = ["ising"],
):
    """Plot ground state energy for Ising model."""
    assert len(model_type) == len(sizes), (
        "model_type and sizes must have the same length"
    )
    for size, start, end, granularity, model in zip(
        sizes, starts, ends, granularities, model_type
    ):
        print(
            f"solving ground state of {model} model with size {size} from {start} to {end} with granularity {granularity}"
        )
        num_points = int((end - start) / granularity) + 2
        bond_lengths = np.round(np.linspace(start, end, num_points), decimals=3)
        # Get operators
        if model == "ising":
            op_gen = load_sparseOp_with_id(bond_lengths, size)
        elif model == "heisenberg":
            op_gen = load_sparseOp_heisenberg_id(bond_lengths, size)
        magnetic_fields, pauli_ops = zip(*op_gen)
        magnetic_fields, pauli_ops = list(magnetic_fields), list(pauli_ops)

        # Calculate ground energies
        ground_energy = []
        print("solving ground state of each Hamiltonian")
        for pauli_op in pauli_ops:
            ground_energy.append(solve_groundstate_numpy(pauli_op).eigenvalue.real)

        # Save results
        energy_bondlength_map = {
            f"{round(float(bond), 2):.2f}": float(energy)
            for bond, energy in zip(magnetic_fields, ground_energy)
        }
        if model == "ising":
            json_file = f"LL-{size}.json"
        elif model == "heisenberg":
            json_file = f"XXZ-{size}.json"
        with open(json_file, "w") as f:
            json.dump(energy_bondlength_map, f, indent=4)

        if plot:
            with plt.style.context(["science", "grid"]):
                plt.figure(figsize=(10, 8))
                plt.plot(bond_lengths, ground_energy)
                plt.xlabel("Magnetic Field")
                plt.ylabel("Ground State Energy (Hartree)")
                plt.xticks(np.linspace(min(bond_lengths), max(bond_lengths), 7))
                if model == "ising":
                    plt.title(f"LL-{size} energy vs magnetic field")
                    plt.savefig(f"LL-{size}.png", dpi=300)
                elif model == "heisenberg":
                    plt.title(f"XXZ-{size} energy vs bond length")
                    plt.savefig(f"XXZ-{size}.png", dpi=300)
                plt.close()


def plot_molecule_energy_from_file(molecule_name: str, slice: np.ndarray):
    with open(f"ground-state/{molecule_name}.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    str_of_slice = [f"{float(bond):.2f}" for bond in slice]
    ground_energy = [data[bd] for bd in str_of_slice]
    # Plot results
    with plt.style.context(["ieee", "grid", "light"]):
        plt.figure(figsize=(10, 8))
        # Increase font size for all text elements
        plt.rcParams.update({"font.size": 14})
        plt.rcParams.update({"axes.labelsize": 16})
        plt.rcParams.update({"axes.titlesize": 18})
        plt.rcParams.update({"xtick.labelsize": 14})
        plt.rcParams.update({"ytick.labelsize": 14})
        plt.rcParams.update({"legend.fontsize": 14})

        plt.plot(slice, ground_energy, linewidth=2.5)  # Make the line thicker
        plt.xlabel("Bond Length (Å)")
        plt.ylabel("Ground-State Energy (Hartree)")
        plt.xticks(np.linspace(min(slice), max(slice), 10))
        if molecule_name == "BeH2":
            plt.title("$BeH_2$ energy vs bond length")
        else:
            plt.title(f"${molecule_name}$ energy vs bond length")
        plt.savefig(
            f"ground-state/{molecule_name}_ground.pdf", dpi=400, bbox_inches="tight"
        )
        plt.close()


if __name__ == "__main__":
    print("start solving ground state of molecules")
    plot_molecule_energy(
        molecule_names=["H2", "LiH", "BeH2", "HF"],
        starts=[0.20, 1.4, 1.2, 0.83],
        ends=[2.50, 1.7, 1.47, 1.2],
        granularities=[0.01, 0.01, 0.01, 0.01],
        plot=True,
    )
    print("start solving ground state of Heisenberg model")
    plot_physical_energy(
        sizes=[5, 5],
        starts=[0.90, 0.90],
        ends=[1.10, 1.10],
        granularities=[0.01, 0.01],
        plot=True,
        model_type=["heisenberg", "ising"],
    )
