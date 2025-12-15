"""Utility functions for TreeVQA applications."""

from typing import List

import numpy as np
import json
import matplotlib.pyplot as plt
import scienceplots
import os
import pickle
import logging


def parse_slice(slice_str: str, precision: int = 3) -> List[List[float]]:
    """Parse slice string into list of float ranges.
    Example:
        >>> parse_slice("0.5:1.5:0.1,2:3:0.2", precision=2)
        [[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4], [2.0, 2.2, 2.4, 2.6, 2.8]]
    """
    result = []
    slice_segments = slice_str.split(",")

    for slice_segment in slice_segments:
        slice_parts = slice_segment.split(":")
        assert len(slice_parts) == 3, (
            f"Invalid slice format: {slice_segment}. Expected format: start:end:step"
        )

        try:
            start, end, step = map(float, slice_parts)
            segment = np.arange(start, end, step)
            rounded_seg = [round(val, precision) for val in segment]
            result.append(rounded_seg)
        except ValueError as e:
            raise ValueError(f"Invalid slice values in {slice_segment}: {e}")

    return result


def concatenate_slices(segment_values: List[List[float]]) -> List[float]:
    """Flatten a list of lists into a single list.
    Example:
        >>> concatenate_slices([[1.0, 1.1], [2.0, 2.1]])
        [1.0, 1.1, 2.0, 2.1]
    """
    return [item for sublist in segment_values for item in sublist]


def visualize_energy_convergence(
    intermediate_data,
    correct_energies,
    result_dir,
    checkpoint_indices,
    optimized_energies,
    optimized_intermediate_data,
) -> None:
    """
    Visualize energy convergence and final accuracy with custom checkpoint indices

    Parameters:
    intermediate_data : np.ndarray (n_checkpoints × n_molecules)
        Matrix containing intermediate energy values
    correct_energies : np.ndarray (n_molecules,)
        Reference ground truth energies
    checkpoint_indices : list[int]
        List of checkpoint step numbers
    result_dir : str
        Output directory for figures
    """
    assert len(checkpoint_indices) == intermediate_data.shape[0], (
        "Checkpoint indices count must match data rows"
    )
    np.save(os.path.join(result_dir, "intermediate_data.npy"), intermediate_data)
    np.save(
        os.path.join(result_dir, "optimized_intermediate_data.npy"),
        np.array(optimized_intermediate_data),
    )
    np.save(os.path.join(result_dir, "checkpoint_indices.npy"), checkpoint_indices)
    np.save(os.path.join(result_dir, "correct_energies.npy"), correct_energies)

    _create_convergence_plot(
        intermediate_data=optimized_intermediate_data,
        correct_energies=correct_energies,
        checkpoint_indices=checkpoint_indices,
        save_dir=result_dir,
    )

    _create_final_accuracy_plot(
        final_energies=optimized_energies,
        correct_energies=correct_energies,
        checkpoint_step=checkpoint_indices[-1],
        save_dir=result_dir,
    )


def save_vqe_intermediate_data(
    intermediate_data,
    checkpoint_indices,
    result_dir,
    op_id,
) -> None:
    """
    Visualize energy convergence and final accuracy with custom checkpoint indices

    Parameters:
    intermediate_data : np.ndarray (n_checkpoints × n_molecules)
        Matrix containing intermediate energy values
    correct_energies : np.ndarray (n_molecules,)
        Reference ground truth energies
    checkpoint_indices : list[int]
        List of checkpoint step numbers
    result_dir : str
        Output directory for figures
    """
    assert len(checkpoint_indices) == intermediate_data.shape[0], (
        "Checkpoint indices count must match data rows"
    )
    intermediate_data = intermediate_data.squeeze()
    molecule_data_file = os.path.join(result_dir, "data.json")

    if os.path.exists(molecule_data_file):
        with open(molecule_data_file, "r") as f:
            molecule_data = json.load(f)
    else:
        molecule_data = {}

    if op_id not in molecule_data.keys():
        molecule_data[op_id] = {}
    for idx, energy in zip(checkpoint_indices, intermediate_data):
        if isinstance(energy, complex):
            molecule_data[op_id][idx] = energy.real
        else:
            molecule_data[op_id][idx] = energy

    try:
        with open(molecule_data_file, "w") as f:
            json.dump(molecule_data, f, indent=4)
    except Exception as e:
        with open("f{molecule_data_file}.pickle", "wb") as f:
            pickle.dump(molecule_data, f)


def _create_convergence_plot(
    intermediate_data, correct_energies, checkpoint_indices, save_dir
):
    """Plot convergence trends with custom checkpoint indices"""
    abs_errors = np.abs(intermediate_data - correct_energies)
    rel_errors = np.abs((intermediate_data - correct_energies) / correct_energies)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(
        checkpoint_indices,
        np.mean(abs_errors, axis=1),
        label="Mean Absolute",
        color="blue",
        marker="o",
    )
    ax1.plot(
        checkpoint_indices,
        np.median(abs_errors, axis=1),
        label="Median Absolute",
        linestyle="--",
        color="green",
    )
    ax1.fill_between(
        checkpoint_indices,
        np.percentile(abs_errors, 25, axis=1),
        np.percentile(abs_errors, 75, axis=1),
        alpha=0.2,
        color="gray",
        label="IQR",
    )

    ax1.set_xlabel("Checkpoint Step", fontsize=12)
    ax1.set_ylabel("Absolute Error", fontsize=12)
    ax1.grid(True, alpha=0.4)
    ax1.set_title("Energy Convergence Trends", fontsize=14)

    ax2 = ax1.twinx()
    ax2.plot(
        checkpoint_indices,
        np.mean(rel_errors, axis=1) * 100,
        color="purple",
        linestyle="-.",
        marker="^",
        label="Mean Relative Error",
    )
    ax2.set_ylabel("Relative Error (%)", fontsize=12)

    lines = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax1.legend(lines, labels, loc="upper right")

    plt.savefig(
        os.path.join(save_dir, "convergence_trends.png"), bbox_inches="tight", dpi=300
    )
    plt.close()


def _create_final_accuracy_plot(
    final_energies, correct_energies, checkpoint_step, save_dir
):
    with plt.style.context("science", "sactter"):
        """Plot final accuracy with metrics annotation"""
        abs_errors = np.abs(final_energies - correct_energies)
        rel_errors = np.abs((final_energies - correct_energies) / correct_energies)

        mean_abs = np.mean(abs_errors)
        mean_rel = np.mean(rel_errors) * 100
        max_abs = np.max(abs_errors)

        fig, ax = plt.subplots(figsize=(8, 8))

        ax.scatter(
            correct_energies,
            final_energies,
            alpha=0.6,
            edgecolor="w",
            c=np.log10(abs_errors + 1e-8),
            cmap="viridis",
        )
        ax.plot(
            [correct_energies.min(), correct_energies.max()],
            [correct_energies.min(), correct_energies.max()],
            "r--",
            lw=1,
            label="Perfect Accuracy",
        )

        ax.set(
            xlabel="Reference Energy",
            ylabel="Computed Energy",
            title=f"Final Accuracy at Checkpoint {checkpoint_step}\n"
            f"Mean Absolute: {mean_abs:.2e}, Mean Relative: {mean_rel:.3f}%",
            aspect="equal",
        )
        ax.grid(alpha=0.3)
        logging.info(f"Mean Absolute Error: {mean_abs:.2e}")
        logging.info(f"Mean Relative Error: {mean_rel:.4f}%")

        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label("log10(Absolute Error)")

        text = (
            f"Mean Absolute Error: {mean_abs:.2e}\n"
            f"Mean Relative Error: {mean_rel:.1f}%\n"
            f"Max Absolute Error: {max_abs:.2e}"
        )
        ax.text(
            0.05,
            0.95,
            text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox=dict(facecolor="white", alpha=0.8),
        )

        save_path = os.path.join(save_dir, "final_accuracy.png")
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()
