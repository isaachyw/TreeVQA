"""
Helper functions for molecule and Hamiltonian processing.

This module provides utilities for:
- Molecular system setup (coordinates, Hamiltonians)
- Pauli string manipulation and averaging
- Ground state computation
"""

import math
import re
import logging
from typing import List, Iterable, Tuple, Dict, Optional, Callable, NewType
from collections import defaultdict

import numpy as np
from scipy.linalg import norm
from scipy.sparse.linalg import eigsh
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit_algorithms import NumPyMinimumEigensolverResult
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper, JordanWignerMapper
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer

from .op_task import OpTask

# Type alias for Hamiltonian info (coefficients, Pauli strings)
Hinfo = NewType("Hinfo", Tuple[Iterable[float], Iterable[str]])

# =============================================================================
# Molecular Configuration Constants
# =============================================================================

MINIMAL_FOCK_SPACE: Dict[str, Tuple[int, int]] = {
    "H2": (2, 2),
    "LiH": (2, 2),
    "BeH2": (4, 4),
    "H2O": (4, 4),
    "HF": (2, 2),
}
"""Minimal active space (electrons, orbitals) for supported molecules."""

FULL_FOCK_SPACE: Dict[str, Tuple[int, int]] = {
    "H2": (2, 2),
    "LiH": (4, 6),
    "BeH2": (6, 7),
    "H2O": (10, 7),
    "HF": (10, 6),
}
"""Full active space (electrons, orbitals) for supported molecules."""


# =============================================================================
# Molecule Coordinate Helpers
# =============================================================================


def get_molecule_coords(molecule_name: str, bond_len: float) -> str:
    """Get PySCF coordinate string for supported molecules.

    Args:
        molecule_name: Name of the molecule (H2, LiH, HF, BeH2, H2O, C2H2).
        bond_len: Bond length in Angstroms.

    Returns:
        PySCF-format coordinate string.

    Raises:
        ValueError: If molecule is not supported.
    """
    coords = {
        "H2": f"H 0.0 0.0 0.0; H {bond_len} 0.0 0.0",
        "LiH": f"Li 0.0 0.0 0.0; H {bond_len} 0.0 0.0",
        "HF": f"F 0.0 0.0 0.0; H {bond_len} 0.0 0.0",
        "BeH2": f"Be 0.0 0.0 0.0; H {bond_len} 0.0 0.0; H {-bond_len} 0.0 0.0",
    }

    if molecule_name in coords:
        return coords[molecule_name]

    if molecule_name == "H2O":
        angle = math.radians(104.5)
        return (
            f"O 0.0 0.0 0.0; "
            f"H {bond_len} 0.0 0.0; "
            f"H {bond_len * math.cos(angle)} {bond_len * math.sin(angle)} 0.0"
        )

    if molecule_name == "C2H2":
        cc = 0.5 * bond_len
        return (
            f"C {-0.5 * cc} 0.0 0.0; C {0.5 * cc} 0.0 0.0; "
            f"H {-0.5 * cc - 0.5647} 0.9288 0.0; H {-0.5 * cc - 0.5647} -0.9288 0.0; "
            f"H {0.5 * cc + 0.5647} 0.9288 0.0; H {0.5 * cc + 0.5647} -0.9288 0.0"
        )

    raise ValueError(f"Unsupported molecule: {molecule_name}")


def format_molecule_string(symbols: List[str], coordinates: List[List[float]]) -> str:
    """Convert symbols and coordinates to PySCF format.

    Args:
        symbols: List of atomic symbols.
        coordinates: List of [x, y, z] coordinates.

    Returns:
        PySCF-format coordinate string.
    """
    return (
        "; ".join(
            f"{sym} {' '.join(str(x) for x in coord)}"
            for sym, coord in zip(symbols, coordinates)
        )
        + "; "
    )


# =============================================================================
# Hamiltonian Computation
# =============================================================================


def molecule_to_op(
    atom_string: str,
    unit: DistanceUnit = DistanceUnit.BOHR,
    new_num_orbitals: Optional[int] = None,
    freeze_core: bool = False,
) -> Tuple[np.ndarray, List, str]:
    """Compute qubit Hamiltonian using Jordan-Wigner mapper.

    Args:
        atom_string: PySCF-format atom coordinate string.
        unit: Distance unit (default: BOHR).
        new_num_orbitals: Active space orbital count (optional).
        freeze_core: If True, freeze core orbitals.

    Returns:
        Tuple of (coefficients, paulis, hf_bitstring).
    """
    logging.disable(logging.INFO)
    try:
        driver = PySCFDriver(
            atom=atom_string, basis="sto3g", charge=0, spin=0, unit=unit
        )
        problem = driver.run()
        mapper = JordanWignerMapper()

        if freeze_core or new_num_orbitals is not None:
            num_electrons = (problem.num_alpha, problem.num_beta)
            if new_num_orbitals is None and freeze_core:
                new_num_orbitals = problem.num_spatial_orbitals - 1
            transformer = ActiveSpaceTransformer(num_electrons, new_num_orbitals)
            problem = transformer.transform(problem)

        qubitOp = mapper.map(problem.hamiltonian.second_q_op())
        hf_state = HartreeFock(
            problem.num_spatial_orbitals, problem.num_particles, mapper
        )

        # Reverse for qiskit endianness
        qubitOp.paulis = [x[::-1] for x in qubitOp.paulis]
        qubitOp.coeffs[0] += problem.nuclear_repulsion_energy
        qubitOp.coeffs = np.array(qubitOp.coeffs).real

        return qubitOp.coeffs, qubitOp.paulis, hf_state._bitstr
    finally:
        logging.disable(logging.NOTSET)


def getHinfo(
    atom_string: str,
    calculate_hf: bool = False,
    ansatz: Optional[QuantumCircuit] = None,
) -> Tuple[Hinfo, np.ndarray]:
    """Compute qubit Hamiltonian using Parity mapper.

    Args:
        atom_string: PySCF-format atom coordinate string.
        calculate_hf: If True, compute Hartree-Fock bitstring.
        ansatz: Required if calculate_hf is True.

    Returns:
        Tuple of (hinfo, hf_bitstring).

    Raises:
        ValueError: If calculate_hf is True but ansatz is None.
    """
    if calculate_hf and ansatz is None:
        raise ValueError("Ansatz required for Hartree-Fock calculation")

    driver = PySCFDriver(
        atom=atom_string, basis="sto3g", charge=0, spin=0, unit=DistanceUnit.ANGSTROM
    )
    problem = driver.run()
    mapper = ParityMapper(num_particles=problem.num_particles)

    qubitOp = mapper.map(problem.hamiltonian.second_q_op())
    paulis = [x[::-1] for x in qubitOp.paulis.to_labels()]
    coeffs = list(qubitOp.coeffs)

    # Add nuclear repulsion to identity term
    idx = paulis.index("I" * len(paulis[0]))
    coeffs[idx] += problem.nuclear_repulsion_energy

    hf_bitstring = np.array([])
    if calculate_hf:
        hf_state = HartreeFock(
            problem.num_spatial_orbitals, problem.num_particles, mapper
        )
        hf_bitstring = np.array([int(x) for x in hf_state._bitstr])

    return (np.array(coeffs).real, np.array(paulis)), hf_bitstring


def uccsd_ansatz(name: str, freeze: bool = False, reps: int = 1) -> UCCSD:
    """Create UCCSD ansatz for supported molecules.

    Args:
        name: Molecule name.
        freeze: If True, use minimal active space.
        reps: Number of UCCSD repetitions.

    Returns:
        UCCSD circuit ansatz.

    Raises:
        ValueError: If molecule is not supported.
    """
    if name not in FULL_FOCK_SPACE:
        raise ValueError(f"Unsupported molecule: {name}")

    fock_map = MINIMAL_FOCK_SPACE if freeze else FULL_FOCK_SPACE
    active_electron, active_orbitals = fock_map[name]

    alpha_spin = (active_electron + 1) // 2
    beta_spin = active_electron // 2

    return UCCSD(
        num_spatial_orbitals=active_orbitals,
        num_particles=(alpha_spin, beta_spin),
        qubit_mapper=JordanWignerMapper(),
        reps=reps,
    )


# =============================================================================
# Ground State Computation
# =============================================================================


def solve_groundstate_numpy(spOp: SparsePauliOp) -> NumPyMinimumEigensolverResult:
    """Solve for ground state using sparse eigenvalue solver.

    Args:
        spOp: Sparse Pauli operator representing the Hamiltonian.

    Returns:
        Result containing eigenvalue and eigenstate.
    """
    matrix = spOp.to_matrix(sparse=True)
    eigenvalues, eigenvectors = eigsh(matrix, k=1, which="SA")

    result = NumPyMinimumEigensolverResult()
    result.eigenvalue = eigenvalues[0]
    result.eigenstate = eigenvectors[:, 0]
    return result


# =============================================================================
# Pauli String Utilities
# =============================================================================


def pennylane_terms_to_qiskit(term: str, num_qubits: int) -> str:
    """Convert Pennylane Pauli term format to Qiskit label.

    Args:
        term: Pennylane-format Pauli string.
        num_qubits: Number of qubits.

    Returns:
        Qiskit-format Pauli label.
    """
    term = term.strip()
    if term == "I()":
        return "I" * num_qubits

    label = ["I"] * num_qubits
    for pauli, qubit_str in re.findall(r"([XYZI])\((\d+)\)", term):
        label[int(qubit_str)] = pauli
    return "".join(label)


def compare_pauli_string(pauli_a: Iterable[str], pauli_b: Iterable[str]) -> bool:
    """Check if two Pauli string collections have the same content."""
    return set(pauli_a) == set(pauli_b)


def sparse_pauli_similarity(pauli_a: SparsePauliOp, pauli_b: SparsePauliOp) -> float:
    """Compute cosine similarity between operator coefficients.

    Args:
        pauli_a: First operator.
        pauli_b: Second operator (must have same Paulis).

    Returns:
        Cosine similarity in [-1, 1].
    """
    assert pauli_a.paulis == pauli_b.paulis
    return np.dot(pauli_a.coeffs, pauli_b.coeffs) / (
        np.linalg.norm(pauli_a.coeffs) * np.linalg.norm(pauli_b.coeffs)
    )


def sparse_pauli_to_matrix(h: Hinfo) -> np.ndarray:
    """Convert Pauli representation to dense matrix.

    Args:
        h: Tuple of (coefficients, pauli_strings).

    Returns:
        Dense Hamiltonian matrix.
    """
    coeffs, paulis = h
    return SparsePauliOp(data=paulis, coeffs=coeffs).to_matrix(sparse=False)


# =============================================================================
# Hamiltonian Similarity Metrics
# =============================================================================


def hamiltonian_similarity(H1: np.ndarray, H2: np.ndarray) -> Dict[str, float]:
    """Compute similarity metrics between two Hermitian matrices.

    Args:
        H1: First Hamiltonian matrix.
        H2: Second Hamiltonian matrix.

    Returns:
        Dictionary with similarity metrics.

    Raises:
        ValueError: If matrices are not square, same size, or Hermitian.
    """
    if H1.shape != H2.shape or H1.shape[0] != H1.shape[1]:
        raise ValueError("Hamiltonians must be square matrices of same size")
    if not np.allclose(H1, H1.conj().T) or not np.allclose(H2, H2.conj().T):
        raise ValueError("Hamiltonians must be Hermitian")

    diff = H1 - H2
    eigenvalues_H1, eigvecs_H1 = np.linalg.eigh(H1)
    eigenvalues_H2, eigvecs_H2 = np.linalg.eigh(H2)

    return {
        "frobenius_norm": norm(diff, "fro"),
        "eigenvalue_difference_norm": norm(eigenvalues_H1 - eigenvalues_H2),
        "trace_distance": np.sum(np.abs(np.linalg.eigvals(diff))),
        "commutator_norm": norm(H1 @ H2 - H2 @ H1, "fro"),
        "average_eigenvector_overlap": np.mean(
            np.abs(np.sum(np.conj(eigvecs_H1) * eigvecs_H2, axis=0)) ** 2
        ),
    }


# =============================================================================
# Operator Averaging
# =============================================================================


def average_pauli_hamiltonians(
    hamiltonians: List[Hinfo], avg_func: Callable
) -> Tuple[Hinfo, List[Hinfo]]:
    """Average Hamiltonians by Pauli string coefficients.

    Args:
        hamiltonians: List of (coeffs, paulis) tuples.
        avg_func: Averaging function (e.g., np.mean).

    Returns:
        Tuple of (averaged_hinfo, padded_hamiltonians).

    Raises:
        ValueError: If hamiltonians list is empty or has mismatched lengths.
    """
    if not hamiltonians:
        raise ValueError("Empty Hamiltonian list")

    # Collect all Pauli strings
    pauli_superset = set()
    for _, pauli_strs in hamiltonians:
        pauli_superset.update(pauli_strs)

    num_hamiltonians = len(hamiltonians)
    total_terms: Dict[str, List[float]] = defaultdict(list)

    # Accumulate coefficients
    for idx, (coeffs, pauli_strs) in enumerate(hamiltonians):
        coeffs, pauli_strs = list(coeffs), list(pauli_strs)
        if len(coeffs) != len(pauli_strs):
            raise ValueError(f"Mismatched lengths at index {idx}")

        coeff_map = dict(zip(pauli_strs, coeffs))
        for pauli_str in pauli_superset:
            total_terms[pauli_str].append(coeff_map.get(pauli_str, 0))

    # Compute averages
    avg_coeffs = [avg_func(total_terms[p]) for p in total_terms]
    pauli_strings = list(total_terms.keys())

    # Build padded Hamiltonians
    padded = [
        ([total_terms[p][i] for p in total_terms], pauli_strings)
        for i in range(num_hamiltonians)
    ]

    return (avg_coeffs, pauli_strings), padded


def average_sparse_pauli(
    ops: List[SparsePauliOp], avg_func: Callable = np.mean
) -> SparsePauliOp:
    """Average SparsePauliOp operators.

    Args:
        ops: List of SparsePauliOp to average.
        avg_func: Averaging function (default: np.mean).

    Returns:
        Averaged SparsePauliOp.
    """
    hinfos = [(op.coeffs, op.paulis) for op in ops]
    avg_hinfo, _ = average_pauli_hamiltonians(hinfos, avg_func)
    return SparsePauliOp(data=avg_hinfo[1], coeffs=avg_hinfo[0])


def average_op_task(ops: List[OpTask], avg_func: Callable = np.mean) -> OpTask:
    """Average OpTask operators.

    Args:
        ops: List of OpTask to average.
        avg_func: Averaging function (default: np.mean).

    Returns:
        Averaged OpTask with op_id=-1.
    """
    hinfos = [(op.coeffs, op.paulis) for op in ops]
    avg_hinfo, _ = average_pauli_hamiltonians(hinfos, avg_func)
    return OpTask(data=avg_hinfo[1], coeffs=avg_hinfo[0], op_id=-1)


# =============================================================================
# Pauli Truncation
# =============================================================================


def truncate_pauli_string(
    avg_info: SparsePauliOp, op_list: List[SparsePauliOp], ratio: float = 0.002
) -> List[SparsePauliOp]:
    """Truncate small Pauli terms based on coefficient ratio.

    Args:
        avg_info: Average operator for determining truncation.
        op_list: Operators to truncate.
        ratio: Fraction of total coefficient weight to truncate.

    Returns:
        List of truncated operators.

    Raises:
        ValueError: If ratio is not in [0, 1].
    """
    if not 0 <= ratio <= 1:
        raise ValueError("Ratio must be in [0, 1]")

    truncate_budget = np.sum(np.abs(avg_info.coeffs)) * ratio
    sorted_indices = np.argsort(np.abs(avg_info.coeffs))
    sorted_avg = avg_info[sorted_indices]

    truncated_paulis = set()
    budget_used = 0.0
    for coeff, pauli in zip(sorted_avg.coeffs, sorted_avg.paulis):
        budget_used += np.abs(coeff)
        if budget_used > truncate_budget:
            break
        truncated_paulis.add(pauli)

    logging.info("Truncated %d of %d Pauli terms", len(truncated_paulis), avg_info.size)

    return [
        SparsePauliOp(
            op.paulis[[p not in truncated_paulis for p in op.paulis]],
            coeffs=op.coeffs[[p not in truncated_paulis for p in op.paulis]],
        )
        for op in op_list
    ]


def get_per_sparse_value(op: SparsePauliOp, pauli_values: Dict[Pauli, float]) -> float:
    """Get expectation value of a sparse operator from per-Pauli values.

    Args:
        op: Sparse Pauli operator.
        pauli_values: Dictionary mapping Pauli terms to expectation values.

    Returns:
        The operator expectation value.
    """
    values = [pauli_values[Pauli(p)] for p in op.paulis]
    return np.dot(values, op.coeffs).real
