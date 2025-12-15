"""Ising and Heisenberg model Hamiltonian generators."""

from typing import Tuple, Generator, Iterable

from qiskit_nature.second_q.hamiltonians.lattices import (
    LineLattice,
    BoundaryCondition,
    SquareLattice,
    Lattice,
)
from qiskit_nature.second_q.hamiltonians import IsingModel, HeisenbergModel
from qiskit_nature.second_q.mappers import LinearMapper
import qiskit_nature.second_q.mappers as mp
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info import Pauli
import numpy as np


def load_sparseOp_with_id(
    MAGNETIC_FIELD: Iterable, num_node: int
) -> Generator[Tuple[float, SparsePauliOp], None, None]:
    """Generate Ising model Hamiltonians for different magnetic fields."""
    mapper = mp.LinearMapper()
    line_lattice = LineLattice(
        num_nodes=num_node, boundary_condition=BoundaryCondition.OPEN
    )
    for magnetic_field in MAGNETIC_FIELD:
        ising_model = IsingModel(
            line_lattice.uniform_parameters(
                uniform_interaction=-1.0,
                uniform_onsite_potential=-magnetic_field,
            ),
        )
        f_op = ising_model.second_q_op()
        sop = mapper.map(f_op)
        yield magnetic_field, sop.sort()


def load_sparseOp_heisenberg_id(
    MAGNETIC_FIELD: Iterable, num_node: int
) -> Generator[Tuple[float, SparsePauliOp], None, None]:
    """Generate Heisenberg model Hamiltonians for different magnetic fields."""
    mapper = mp.LinearMapper()
    line_lattice = LineLattice(
        num_nodes=num_node, boundary_condition=BoundaryCondition.OPEN
    )
    for magnetic_field in MAGNETIC_FIELD:
        heisenberg_model = HeisenbergModel(
            line_lattice, (1.0, 1.0, magnetic_field), (0, 0, 0)
        )
        f_op = heisenberg_model.second_q_op()
        sop = mapper.map(f_op)
        yield magnetic_field, sop.sort()


def load_sparseOp_with_id_square(
    MAGNETIC_FIELD: Iterable, num_node: int
) -> Generator[Tuple[float, SparsePauliOp], None, None]:
    """Generate square lattice Ising model Hamiltonians."""
    mapper = mp.LinearMapper()
    square_lattice = SquareLattice(
        rows=num_node, cols=num_node, boundary_condition=BoundaryCondition.OPEN
    )
    for magnetic_field in MAGNETIC_FIELD:
        ising_model = IsingModel(
            square_lattice.uniform_parameters(
                uniform_interaction=-1.0,
                uniform_onsite_potential=-magnetic_field,
            ),
        )
        f_op = ising_model.second_q_op()
        sop = mapper.map(f_op)
        yield magnetic_field, sop.sort()


if __name__ == "__main__":
    num = 5
    mag_field = [1.2]
    list(load_sparseOp_with_id(mag_field, num))
