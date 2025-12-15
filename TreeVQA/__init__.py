"""
TreeVQA - A Tree-Structured Execution Framework for Shot Reduction in Variational Quantum Algorithms


Main Components:
    - TreeVQA: Main orchestrator for parallel VQE optimization
    - VQECluster: Single VQE instance with TreeVQA support
    - OpTask: Operator container with tracking capabilities
    - SPSAP/COBYLAP: Optimizers with TreeVQA integration
"""

from typing import List

# Core TreeVQA components
from .TreeVQA import TreeVQA, TreeVQAConfig, TreeVQAResult, plot_overview
from .op_task import OpTask

# VQA components
from .vqa import VQECluster, VQEClusterResult, SegmentEnergy

# Optimizers
from .optimizer import SPSAP, SPSAHyperParams, COBYLAP, COBYLAPConfig
from .optimizer import TreeVQA_Yielder

# Helper functions
from .treevqa_helper import (
    average_op_task,
    average_sparse_pauli,
    solve_groundstate_numpy,
    molecule_to_op,
    get_molecule_coords,
    uccsd_ansatz,
)

# Application experiments
from .application import (
    MoleculeExperiment,
    MaxCutExperiment,
    IsingModelExperiment,
    PowerExperiment,
)

__all__: List[str] = [
    "TreeVQA",
    "TreeVQAConfig",
    "TreeVQAResult",
    "OpTask",
    "VQECluster",
    "VQEClusterResult",
    "SegmentEnergy",
    # Optimizers
    "SPSAP",
    "SPSAHyperParams",
    "COBYLAP",
    "COBYLAPConfig",
    "TreeVQA_Yielder",
    # Helpers
    "average_op_task",
    "average_sparse_pauli",
    "solve_groundstate_numpy",
    "molecule_to_op",
    "get_molecule_coords",
    "uccsd_ansatz",
    "plot_overview",
    # Applications
    "MoleculeExperiment",
    "MaxCutExperiment",
    "IsingModelExperiment",
    "PowerExperiment",
]

__version__ = "0.1.0"
