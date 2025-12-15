"""
Noise model configuration for quantum simulations.

This module provides utilities for creating noise models from IBM fake
backends or custom error specifications.
"""

from collections import OrderedDict
from typing import Optional

from numpy.random import normal
from qiskit_aer.noise import NoiseModel, pauli_error
from qiskit_ibm_runtime.fake_provider import (
    FakeCairoV2,
    FakeHanoiV2,
    FakeMumbaiV2,
    FakeOsaka,
    FakeKolkataV2,
    FakeAuckland,
)
from qiskit.converters import circuit_to_dag, dag_to_circuit

# =============================================================================
# Available Fake Backends
# =============================================================================

FAKE_BACKENDS = {
    "FakeCairoV2": FakeCairoV2,
    "FakeHanoiV2": FakeHanoiV2,
    "FakeMumbaiV2": FakeMumbaiV2,
    "FakeOsaka": FakeOsaka,
    "FakeKolkataV2": FakeKolkataV2,
    "FakeAuckland": FakeAuckland,
}


# =============================================================================
# Noise Model Construction
# =============================================================================


def build_model(
    reset_error: float,
    measure_error: float,
    gate_error: float,
    T1: Optional[float] = None,
    T2: Optional[float] = None,
) -> NoiseModel:
    """Build a custom noise model with Pauli errors.

    Args:
        reset_error: Error probability for reset operations.
        measure_error: Error probability for measurements.
        gate_error: Error probability for two-qubit gates.
        T1: T1 relaxation time in nanoseconds (randomly sampled if None).
        T2: T2 dephasing time in nanoseconds (randomly sampled if None).

    Returns:
        Configured NoiseModel with specified error rates.
    """
    if T1 is None:
        T1 = normal(50e3, 10e3)
    if T2 is None:
        T2 = normal(70e3, 10e3)
    T2 = min(T2, 2 * T1)  # Physical constraint: T2 <= 2*T1

    error_reset = pauli_error([("X", reset_error), ("I", 1 - reset_error)])
    error_meas = pauli_error([("X", measure_error), ("I", 1 - measure_error)])
    error_gate1 = pauli_error([("X", gate_error / 10), ("I", 1 - gate_error / 10)])
    error_gate2_base = pauli_error([("X", gate_error), ("I", 1 - gate_error)])
    error_gate2 = error_gate2_base.tensor(error_gate2_base)

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_reset, "reset")
    noise_model.add_all_qubit_quantum_error(error_meas, "measure")
    noise_model.add_all_qubit_quantum_error(error_gate1, ["id", "x", "sx", "rz"])
    noise_model.add_all_qubit_quantum_error(error_gate2, ["cx"])

    return noise_model


def build_device_noise_model(backend_name: str) -> NoiseModel:
    """Build a noise model from a fake IBM backend.

    Args:
        backend_name: Name of the fake backend (e.g., "FakeKolkataV2").

    Returns:
        NoiseModel derived from the specified backend.

    Raises:
        ValueError: If backend_name is not in FAKE_BACKENDS.
    """
    if backend_name not in FAKE_BACKENDS:
        raise ValueError(
            f"Unknown backend: {backend_name}. Available: {list(FAKE_BACKENDS.keys())}"
        )
    backend = FAKE_BACKENDS[backend_name]()
    return NoiseModel.from_backend(backend)


# =============================================================================
# Circuit Utilities
# =============================================================================


def remove_idle_qwires(circ):
    """Remove idle quantum wires from a circuit.

    Useful for optimizing circuits where some qubits are unused.

    Args:
        circ: Quantum circuit to modify.

    Returns:
        New circuit with idle wires removed.
    """
    dag = circuit_to_dag(circ)
    idle_wires = list(dag.idle_wires())
    for w in idle_wires:
        dag._remove_idle_wire(w)
        dag.qubits.remove(w)
    dag.qregs = OrderedDict()
    return dag_to_circuit(dag)
