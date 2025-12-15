"""
Utility functions for interfacing between Qiskit and PauliPropagation.jl.

This module provides functions to convert between Qiskit quantum operators
and PauliPropagation.jl data structures using juliacall for efficient
simulation of large quantum circuits.
"""

import time
from typing import Optional, Callable, Dict

import numpy as np
from scipy.optimize import minimize, OptimizeResult
from qiskit.quantum_info import SparsePauliOp, Pauli
from juliacall import Main as jl

# =============================================================================
# Julia Initialization
# =============================================================================
# check if the PauliPropagation.jl package is installed, if not, install it
if not jl.seval("using PauliPropagation"):
    jl.seval('Pkg.add("PauliPropagation")')

jl.seval("using PauliPropagation")

_pp = jl.PauliPropagation
_pauli_symbols = {
    "I": jl.Symbol("I"),
    "X": jl.Symbol("X"),
    "Y": jl.Symbol("Y"),
    "Z": jl.Symbol("Z"),
}

# Define Julia noise gate types
jl.seval("""
function applynoiselayer(
    psum::PauliPropagation.PauliSum;
    depol_strength=0.02,
    dephase_strength=0.02,
    noise_level=1.0,
)
    for (pstr, coeff) in psum
        new_coeff = coeff *
            (1 - noise_level * depol_strength) ^ PauliPropagation.countweight(pstr) *
            (1 - noise_level * dephase_strength) ^ PauliPropagation.countxy(pstr)
        PauliPropagation.set!(psum, pstr, new_coeff)
    end
    return psum
end

struct NoiseLayer <: PauliPropagation.Gate
    nqubits::Int
    depol_strength::Float64
    dephase_strength::Float64
    noise_level::Float64
end

NoiseLayer(nqubits::Int; depol_strength=0.001, dephase_strength=0.001, noise_level=1.0) = 
    NoiseLayer(nqubits, depol_strength, dephase_strength, noise_level)

function PauliPropagation.propagate(gate::NoiseLayer, psum::PauliPropagation.PauliSum, params, param_count)
    applynoiselayer(psum, depol_strength=gate.depol_strength, 
                    dephase_strength=gate.dephase_strength, 
                    noise_level=gate.noise_level)
    return psum, param_count
end

PauliPropagation.countparameters(gate::NoiseLayer) = 0

struct SingleQubitDepolarizing <: PauliPropagation.Gate
    qubit::Int
    depol_strength::Float64
end

SingleQubitDepolarizing(qubit::Int; depol_strength=0.001) = SingleQubitDepolarizing(qubit, depol_strength)

function PauliPropagation.apply(gate::SingleQubitDepolarizing, pstr::PauliPropagation.PauliStringType, coeff, theta)
    pauli_on_qubit = PauliPropagation.getpauli(pstr, gate.qubit)
    if pauli_on_qubit != :I
        new_coeff = coeff * (1 - (4/3) * gate.depol_strength)
        return ((pstr, new_coeff),)
    else
        return ((pstr, coeff),)
    end
end

PauliPropagation.countparameters(gate::SingleQubitDepolarizing) = 0

struct TwoQubitDepolarizing <: PauliPropagation.Gate
    qubit1::Int
    qubit2::Int
    depol_strength::Float64
end

TwoQubitDepolarizing(qubit1::Int, qubit2::Int; depol_strength=0.001) = TwoQubitDepolarizing(qubit1, qubit2, depol_strength)

function PauliPropagation.apply(gate::TwoQubitDepolarizing, pstr::PauliPropagation.PauliStringType, coeff, theta)
    weight = 0
    if PauliPropagation.getpauli(pstr, gate.qubit1) != :I
        weight += 1
    end
    if PauliPropagation.getpauli(pstr, gate.qubit2) != :I
        weight += 1
    end
    if weight > 0
        new_coeff = coeff * (1 - (4/3) * gate.depol_strength) ^ weight
        return ((pstr, new_coeff),)
    else
        return ((pstr, coeff),)
    end
end

PauliPropagation.countparameters(gate::TwoQubitDepolarizing) = 0
""")


# =============================================================================
# Qiskit <-> Julia Conversion
# =============================================================================


def qiskit_to_paulisum(
    sparse_pauli_op: SparsePauliOp,
) -> "jl.PauliPropagation.PauliSum":
    """Convert a Qiskit SparsePauliOp to a PauliPropagation.jl PauliSum.

    Args:
        sparse_pauli_op: The Qiskit SparsePauliOp to convert.

    Returns:
        PauliPropagation.jl PauliSum object equivalent to the input.
    """
    nqubits = sparse_pauli_op.num_qubits
    pauli_strings = []

    for pauli, coeff in zip(sparse_pauli_op.paulis, sparse_pauli_op.coeffs):
        pauli_str = pauli.to_label()
        symbols = []
        positions = []

        for i, pauli_char in enumerate(pauli_str):
            if pauli_char != "I":
                symbols.append(_pauli_symbols[pauli_char])
                positions.append(i + 1)  # 1-based indexing for Julia

        if len(symbols) == 0:
            pauli_string = _pp.PauliString(nqubits, _pauli_symbols["I"], 0, coeff)
        elif len(symbols) == 1:
            pauli_string = _pp.PauliString(nqubits, symbols[0], positions[0], coeff)
        else:
            pauli_string = _pp.PauliString(nqubits, symbols, positions, coeff)

        pauli_strings.append(pauli_string)

    if len(pauli_strings) == 0:
        zero_pauli = _pp.PauliString(nqubits, 0.0)
        return _pp.PauliSum(zero_pauli)
    elif len(pauli_strings) == 1:
        return _pp.PauliSum(pauli_strings[0])
    else:
        pauli_sum = _pp.PauliSum(pauli_strings[0])
        for pstr in pauli_strings[1:]:
            pauli_sum = pauli_sum + _pp.PauliSum(pstr)
        return pauli_sum


def paulisum_to_qiskit(pauli_sum: "jl.PauliPropagation.PauliSum") -> SparsePauliOp:
    """Convert a PauliPropagation.jl PauliSum to a Qiskit SparsePauliOp.

    Args:
        pauli_sum: The PauliPropagation.jl PauliSum to convert.

    Returns:
        Qiskit SparsePauliOp equivalent to the input.
    """
    nqubits = int(jl.seval(f"length({pauli_sum}.paulis[1].ops)"))
    pauli_list = []
    coeffs = []

    num_terms = len(pauli_sum)

    for i in range(num_terms):
        pauli_string = jl.seval(f"{pauli_sum}.paulis[{i + 1}]")
        coeff = float(jl.seval(f"{pauli_sum}.coeffs[{i + 1}]"))

        pauli_chars = ["I"] * nqubits
        ops = jl.seval(f"{pauli_string}.ops")

        for j in range(nqubits):
            op_symbol = str(jl.seval(f"string({ops}[{j + 1}])"))
            pauli_chars[j] = op_symbol

        pauli_str = "".join(pauli_chars)
        pauli_list.append(pauli_str)
        coeffs.append(coeff)

    if len(pauli_list) == 0:
        return SparsePauliOp.from_list([("I" * nqubits, 0.0)])

    pauli_data = list(zip(pauli_list, coeffs))
    return SparsePauliOp.from_list(pauli_data)


# =============================================================================
# Circuit Creation
# =============================================================================


def create_efficientsu2_circuit(
    num_qubits: int, layers: int, noisy: bool = False, depol_strength: float = 0.001
) -> "jl.Vector":
    """Create an efficientsu2 circuit with circular entanglement topology.

    Creates a hardware-efficient circuit consisting of layers of single-qubit
    Y-Z Pauli rotations and CNOT entangling gates arranged in a ring topology.

    Args:
        num_qubits: Number of qubits in the circuit.
        layers: Number of layers in the circuit.
        noisy: If True, applies depolarizing noise after each gate.
        depol_strength: Depolarizing strength for noise gates.

    Returns:
        PauliPropagation.jl circuit object (vector of Gate objects).

    Raises:
        ValueError: If num_qubits < 2 or layers < 1.
    """
    if num_qubits < 2:
        raise ValueError("Number of qubits must be at least 2")
    if layers < 1:
        raise ValueError("Number of layers must be at least 1")

    if noisy:
        func_str = """function efficientsu2_qiskit_noisy(nqubits::Integer, nlayers::Integer; depol_strength=0.001)
            topology = staircasetopology(nqubits, periodic=true)
            circuit::Vector{Gate} = []

            for jj in 1:nlayers
                for ii in 1:nqubits
                    push!(circuit, PauliRotation(:Y, ii))
                    push!(circuit, SingleQubitDepolarizing(ii, depol_strength=depol_strength))
                    push!(circuit, PauliRotation(:Z, ii))
                    push!(circuit, SingleQubitDepolarizing(ii, depol_strength=depol_strength))
                end

                for pair in reverse(topology)
                    push!(circuit, CliffordGate(:CNOT, pair))
                    push!(circuit, TwoQubitDepolarizing(pair[1], pair[2], depol_strength=5*depol_strength))
                end
            end
            
            for ii in 1:nqubits
                push!(circuit, PauliRotation(:Y, ii))
                push!(circuit, SingleQubitDepolarizing(ii, depol_strength=depol_strength))
                push!(circuit, PauliRotation(:Z, ii))
                push!(circuit, SingleQubitDepolarizing(ii, depol_strength=depol_strength))
            end

            return circuit
        end"""
        return jl.seval(func_str)(num_qubits, layers, depol_strength=depol_strength)
    else:
        func_str = """function efficientsu2_qiskit(nqubits::Integer, nlayers::Integer)
            topology = staircasetopology(nqubits, periodic=true)
            circuit::Vector{Gate} = []

            for jj in 1:nlayers
                for ii in 1:nqubits
                    push!(circuit, PauliRotation(:Y, ii))
                    push!(circuit, PauliRotation(:Z, ii))
                end

                for pair in reverse(topology)
                    push!(circuit, CliffordGate(:CNOT, pair))
                end
            end
            
            for ii in 1:nqubits
                push!(circuit, PauliRotation(:Y, ii))
                push!(circuit, PauliRotation(:Z, ii))
            end

            return circuit
        end"""
        return jl.seval(func_str)(num_qubits, layers)


def print_circuit_info(circuit: "jl.Vector", num_qubits: int) -> None:
    """Print information about a PauliPropagation.jl circuit.

    Args:
        circuit: PauliPropagation.jl circuit object.
        num_qubits: Number of qubits in the circuit.
    """
    circuit_length = len(circuit)
    num_params = jl.seval(f"countparameters({circuit})")

    print("Circuit Information:")
    print(f"  Number of qubits: {num_qubits}")
    print(f"  Total gates: {circuit_length}")
    print(f"  Parameters: {num_params}")

    gate_counts = {}
    for i in range(circuit_length):
        gate = jl.seval(f"{circuit}[{i + 1}]")
        gate_type = str(type(gate)).split(".")[-1].rstrip("'>")
        gate_counts[gate_type] = gate_counts.get(gate_type, 0) + 1

    print("  Gate types:")
    for gate_type, count in gate_counts.items():
        print(f"    {gate_type}: {count}")


# =============================================================================
# Expectation Value Estimation
# =============================================================================


def estimate_pp(
    sparse_pauli_op: SparsePauliOp,
    circuit: "jl.Vector",
    parameters: np.ndarray,
    max_freq: Optional[int] = 8,
    max_weight: Optional[int] = 8,
    min_abs_coeff: Optional[float] = 1e-6,
) -> float:
    """Estimate expectation value with a pre-built circuit and parameters.

    Args:
        sparse_pauli_op: The observable operator to estimate.
        circuit: Pre-built PauliPropagation.jl circuit.
        parameters: Circuit parameters.
        max_freq: Maximum frequency for truncation.
        max_weight: Maximum weight for Pauli string truncation.
        min_abs_coeff: Minimum absolute coefficient for truncation.

    Returns:
        The estimated expectation value.
    """
    pauli_sum = qiskit_to_paulisum(sparse_pauli_op)

    propagated_sum = _pp.propagate(
        circuit,
        pauli_sum,
        parameters,
        max_weight=max_weight,
        min_abs_coeff=min_abs_coeff,
        max_freq=max_freq,
    )

    expectation_value = _pp.overlapwithzero(propagated_sum)
    return expectation_value.real


def estimate_pp_per_pauli(
    sparse_pauli_op: SparsePauliOp,
    circuit: "jl.Vector",
    parameters: np.ndarray,
    max_freq: Optional[int] = 8,
    max_weight: Optional[int] = 8,
    min_abs_coeff: Optional[float] = 1e-6,
) -> Dict[Pauli, float]:
    """Estimate expectation value for each Pauli term of the operator.

    Args:
        sparse_pauli_op: The observable operator.
        circuit: Pre-built PauliPropagation.jl circuit.
        parameters: Circuit parameters.
        max_freq: Maximum frequency for truncation.
        max_weight: Maximum weight for Pauli string truncation.
        min_abs_coeff: Minimum absolute coefficient for truncation.

    Returns:
        Dictionary mapping Pauli terms to their estimated expectation values.
    """
    pauli_values: Dict[Pauli, float] = {}
    operator_srt = sparse_pauli_op.sort()

    for pauli in operator_srt.paulis:
        auxilary_spop = SparsePauliOp.from_list([(pauli.to_label(), 1.0)])
        pauli_sum_aux = qiskit_to_paulisum(auxilary_spop)

        propagated_sum = _pp.propagate(
            circuit,
            pauli_sum_aux,
            parameters,
            max_weight=max_weight,
            min_abs_coeff=min_abs_coeff,
            max_freq=max_freq,
        )

        expectation_value = _pp.overlapwithzero(propagated_sum)
        pauli_values[pauli] = expectation_value.real

    return pauli_values


# =============================================================================
# VQE with PauliPropagation
# =============================================================================


def run_vqe_pp(
    ansatz: "jl.Vector",
    hamiltonian: SparsePauliOp,
    maxiter: int = 100,
    callback: Optional[Callable] = None,
    shots: Optional[int] = None,
    method: str = "COBYLA",
    initial_point: Optional[np.ndarray] = None,
    options: Optional[dict] = None,
    max_freq: Optional[int] = 10,
    max_weight: Optional[int] = 10,
    min_abs_coeff: Optional[float] = 1e-8,
) -> OptimizeResult:
    """Run VQE optimization using PauliPropagation.jl for simulation.

    Args:
        ansatz: PauliPropagation.jl circuit ansatz.
        hamiltonian: Hamiltonian operator to minimize.
        maxiter: Maximum number of iterations.
        callback: Optional callback function.
        shots: Shot count (unused, for API compatibility).
        method: Scipy optimization method.
        initial_point: Initial parameters.
        options: Additional scipy options.
        max_freq: Maximum frequency for truncation.
        max_weight: Maximum weight for truncation.
        min_abs_coeff: Minimum coefficient for truncation.

    Returns:
        Scipy OptimizeResult with optimization results.
    """
    if initial_point is None:
        num_params = jl.seval(f"countparameters({ansatz})")
        initial_point = np.zeros(num_params)

    def objective(params, ansatz, hamiltonian, shots):
        return estimate_pp(
            hamiltonian, ansatz, params, max_freq, max_weight, min_abs_coeff
        )

    options = (
        {"maxiter": maxiter} if options is None else {"maxiter": maxiter, **options}
    )

    result = minimize(
        objective,
        initial_point,
        method=method,
        options=options,
        args=(ansatz, hamiltonian, shots),
        callback=callback,
    )
    return result


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    from qiskit_nature.second_q.hamiltonians.lattices import (
        LineLattice,
        BoundaryCondition,
    )
    from qiskit_nature.second_q.hamiltonians import IsingModel
    import qiskit_nature.second_q.mappers as mp

    def load_sparseOp_with_id(magnetic_fields, num_node):
        mapper = mp.LinearMapper()
        line_lattice = LineLattice(
            num_nodes=num_node, boundary_condition=BoundaryCondition.OPEN
        )
        for magnetic_field in magnetic_fields:
            ising_model = IsingModel(
                line_lattice.uniform_parameters(
                    uniform_interaction=-1.0,
                    uniform_onsite_potential=-magnetic_field,
                ),
            )
            f_op = ising_model.second_q_op()
            sop = mapper.map(f_op)
            yield magnetic_field, sop.sort()

    H = list(load_sparseOp_with_id([1.0], 10))[0][1]
    print(f"Hamiltonian num_qubits: {H.num_qubits}")

    circuit = create_efficientsu2_circuit(H.num_qubits, 2)
    start_time = time.time()
    result = run_vqe_pp(circuit, H, maxiter=500, method="COBYLA")
    elapsed = time.time() - start_time

    print(f"Time taken: {elapsed:.2f} seconds")
    print(result)
