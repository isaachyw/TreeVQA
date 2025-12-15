from dataclasses import dataclass
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import scienceplots
from typing import Optional


plt.style.use(["science", "grid"])


@dataclass
class Result_analysis:
    result_dir: str
    checkpoint_indices: np.ndarray  # 1D array of shot (or iteration) checkpoints
    reference_energies: np.ndarray  # 1D array of "true" or reference energies
    intermediate_data: np.ndarray  # (N, M) unoptimized energies at checkpoints
    optimized_intermediated_data: np.ndarray  # (N, M) optimized energies at checkpoints

    # Baseline (separate VQE) info
    seperate_vqe_dir: str
    op_ids: np.ndarray  # Array of molecule/problem identifiers
    seperate_vqe_checkpoint_indices: (
        np.ndarray
    )  # 1D array of baseline shot (or iteration) checkpoints
    seperate_vqe_intermediate_data: (
        np.ndarray
    )  # (N, M) baseline energies at checkpoints
    save_dir: str
    num_pauli_str: int
    num_batch: int
    optimizer_constant: int
    molecule_name: str

    def __init__(
        self,
        result_dir: str,
        seperate_vqe_dir: str,
        save_dir: str,
        num_pauli_str: int,
        num_batch: int,
        optimizer_constant: int,
        molecule_name: str,
    ):
        """
        Parameters
        ----------
        result_dir : str
            Directory containing the main approach's results.
        seperate_vqe_dir : str
            Directory containing the baseline (separate VQE) results.
        """
        self.num_pauli_str = num_pauli_str
        self.num_batch = num_batch
        self.optimizer_constant = optimizer_constant
        self.shot_factor = num_batch * num_pauli_str * optimizer_constant
        self.molecule_name = molecule_name
        # Store input directories
        self.result_dir = result_dir
        self.seperate_vqe_dir = seperate_vqe_dir
        self.save_dir = save_dir
        self.molecule_name = molecule_name
        # Load main approach data
        self.checkpoint_indices = np.load(
            os.path.join(result_dir, "checkpoint_indices.npy")
        )
        self.reference_energies = np.load(
            os.path.join(result_dir, "correct_energies.npy")
        )
        self.intermediate_data = np.load(
            os.path.join(result_dir, "intermediate_data.npy")
        )
        self.optimized_intermediated_data = np.load(
            os.path.join(result_dir, "optimized_intermediate_data.npy")
        )
        self.op_ids = np.load(os.path.join(result_dir, "op_ids.npy"))

        # Load baseline (separate VQE) checkpoint indices as intersection of all keys across op_ids
        molecule_data_file = os.path.join(self.seperate_vqe_dir, "data.json")
        with open(molecule_data_file, "r", encoding="utf-8") as f:
            molecule_data = json.load(f)
        # Get the set of keys for each op_id and find their intersection
        key_sets = [set(molecule_data[str(op_id)].keys()) for op_id in self.op_ids]
        common_keys = set.intersection(*key_sets)
        # Convert to numpy array of integers
        self.seperate_vqe_checkpoint_indices = np.array(list(common_keys)).astype(int)
        # Sort the indices for consistency
        self.seperate_vqe_checkpoint_indices.sort()
        # Load baseline energies for each op_id and stack into shape (num_checkpoints, num_molecules)
        vqe_data_list = []
        for op_id in self.op_ids:
            vqe_dict = molecule_data[str(op_id)]
            try:
                energies = list(
                    [vqe_dict[str(idx)] for idx in self.seperate_vqe_checkpoint_indices]
                )
            except KeyError:
                raise KeyError(f"KeyError: {op_id} not found in {molecule_data}")
            vqe_data_list.append(np.array(energies))
        # Stack along first axis and transpose to shape (num_checkpoints, #op_ids)
        self.seperate_vqe_intermediate_data = np.array(vqe_data_list).T
        self.num_molecules = self.intermediate_data.shape[1]

    @staticmethod
    def compute_fidelity(
        estimated_energies: np.ndarray, reference_energies: np.ndarray
    ) -> np.ndarray:
        """
        Example fidelity definition.
        Adjust if your fidelity metric differs.

        fidelity = 1 - |(E_estimated - E_ref) / E_ref|
        """
        return 1.0 - np.abs(estimated_energies - reference_energies) / np.abs(
            reference_energies
        )

    def visualize_shot_reduction_all_molecules(self, fixed_fidelity: float) -> None:
        """
        1(b). Compare how many molecules exceed the fixed_fidelity for:
              (A) The optimized approach
              (B) The baseline (separate VQE).
        Plot fraction of molecules meeting threshold vs. shots for both.
        Also identify earliest checkpoint (if any) that gets 100% success.
        """
        num_checkpoints_main = len(self.checkpoint_indices)
        num_checkpoints_base = len(self.seperate_vqe_checkpoint_indices)

        # Fraction that meet threshold for main approach
        fractions_main = []
        for idx in range(num_checkpoints_main):
            fidelity_ckpt = self.compute_fidelity(
                self.optimized_intermediated_data[idx, :], self.reference_energies
            )
            frac = np.mean(fidelity_ckpt >= fixed_fidelity)
            fractions_main.append(frac)

        # Fraction that meet threshold for baseline
        fractions_base = []
        for idx in range(num_checkpoints_base):
            fidelity_ckpt_base = self.compute_fidelity(
                self.seperate_vqe_intermediate_data[idx, :], self.reference_energies
            )
            frac_base = np.mean(fidelity_ckpt_base >= fixed_fidelity)
            fractions_base.append(frac_base)

        fractions_main = np.array(fractions_main)
        fractions_base = np.array(fractions_base)

        # Find earliest checkpoint in main approach that gets 100% success
        all_met_indices = np.where(fractions_main == 1.0)[0]
        if len(all_met_indices) > 0:
            earliest_idx_main = all_met_indices[0]
            shots_for_all_main = self.checkpoint_indices[earliest_idx_main]
            print(
                f"[Main Approach] All molecules reach fidelity >= {fixed_fidelity} at checkpoint "
                f"{earliest_idx_main} (Shots = {shots_for_all_main})."
            )
        else:
            print(
                f"[Main Approach] Not all molecules ever reach fidelity >= {fixed_fidelity} "
                "within the given checkpoints."
            )

        # Find earliest checkpoint in baseline that gets 100% success
        all_met_indices_base = np.where(fractions_base == 1.0)[0]
        if len(all_met_indices_base) > 0:
            earliest_idx_base = all_met_indices_base[0]
            shots_for_all_base = self.seperate_vqe_checkpoint_indices[earliest_idx_base]
            print(
                f"[Baseline] All molecules reach fidelity >= {fixed_fidelity} at checkpoint "
                f"{earliest_idx_base} (Shots = {shots_for_all_base})."
            )
        else:
            print(
                f"[Baseline] Not all molecules ever reach fidelity >= {fixed_fidelity} "
                "within the given checkpoints."
            )

        # Plot the fraction vs. shots for both approaches
        plt.figure(figsize=(8, 6))
        plt.plot(
            self.seperate_vqe_checkpoint_indices,
            fractions_base,
            marker="o",
            label="Baseline (Separate VQE)",
        )
        plt.plot(
            self.checkpoint_indices, fractions_main, marker="o", label="Main Approach"
        )
        plt.xlabel("Shots")
        plt.ylabel(f"Fraction of molecules (≥ {fixed_fidelity} fidelity)")
        plt.title(f"All Molecules Shot Reduction (≥ {fixed_fidelity} fidelity)")
        plt.ylim([0, 1.05])
        plt.legend()
        plt.show()

    def visualize_shots_for_various_fidelity_all_molecules(
        self,
        thresholds: list[float],
        truncation: bool = True,
        example_threshold: Optional[float] = None,
    ) -> None:
        """
        For each fidelity threshold in 'thresholds':
          - Determine how many shots the main approach and the baseline each need
            so that *all* molecules exceed that threshold.
          - If it's never reached, store `np.nan`.

        Then plot:
          x-axis = fidelity threshold
          y-axis = shots needed for *all* molecules
          Two lines: main approach vs. baseline

        If the truncation is True, just plot the idx before the example_threshold.
        """

        # Sort thresholds to have a left-to-right plot
        thresholds = sorted(thresholds)

        # Arrays to hold shots needed for each threshold
        shots_needed_main = []
        shots_needed_base = []

        # Pre-compute fidelity for each checkpoint to speed up repeated checks
        # Main approach shape: (num_checkpoints, num_molecules)
        fidelity_main = []
        for i_ckpt in range(len(self.checkpoint_indices)):
            f = self.compute_fidelity(
                self.optimized_intermediated_data[i_ckpt, :], self.reference_energies
            )
            fidelity_main.append(f)
        fidelity_main = np.array(fidelity_main)

        # Baseline shape: (num_checkpoints, num_molecules)
        fidelity_base = []
        for i_ckpt in range(len(self.seperate_vqe_checkpoint_indices)):
            f = self.compute_fidelity(
                self.seperate_vqe_intermediate_data[i_ckpt, :], self.reference_energies
            )
            fidelity_base.append(f)
        fidelity_base = np.array(fidelity_base)

        for T in thresholds:
            # ---- MAIN APPROACH ----
            # Earliest checkpoint where all molecules exceed T
            # i.e. fidelity_main[i_ckpt, :].all() >= T
            # We'll find the first i_ckpt that satisfies this
            # If none do, store np.nan
            found_ckpt_main = np.nan
            for i_ckpt in range(len(self.checkpoint_indices)):
                # Check if all molecules exceed threshold T
                if np.all(fidelity_main[i_ckpt, :] >= T):
                    found_ckpt_main = self.checkpoint_indices[i_ckpt]
                    break
            shots_needed_main.append(found_ckpt_main)

            # ---- BASELINE ----
            found_ckpt_base = np.nan
            for i_ckpt in range(len(self.seperate_vqe_checkpoint_indices)):
                if np.all(fidelity_base[i_ckpt, :] >= T):
                    found_ckpt_base = self.seperate_vqe_checkpoint_indices[i_ckpt]
                    break
            shots_needed_base.append(found_ckpt_base * self.num_molecules)

        # Convert to arrays for plotting
        shots_needed_main = np.array(shots_needed_main)
        shots_needed_base = np.array(shots_needed_base)

        # Find the last valid point for VQE baseline if example_threshold is not None
        if example_threshold is not None:
            last_valid_idx = np.where(np.array(thresholds) <= example_threshold)[0]
        else:
            last_base_idx = np.where(~np.isnan(shots_needed_base))[0]
            last_main_idx = np.where(~np.isnan(shots_needed_main))[0]
            # last_valid_idx is the shorter one
            last_valid_idx = (
                last_base_idx
                if len(last_base_idx) < len(last_main_idx)
                else last_main_idx
            )
        plot_thresholds = thresholds
        plot_shots_needed_main = shots_needed_main
        plot_shots_needed_base = shots_needed_base
        if truncation and len(last_valid_idx) > 0:
            plot_thresholds = thresholds[: last_valid_idx[-1] + 1]
            plot_shots_needed_main = shots_needed_main[: last_valid_idx[-1] + 1]
            plot_shots_needed_base = shots_needed_base[: last_valid_idx[-1] + 1]
        with plt.style.context(("science", "grid")):
            # Create a figure
            plt.figure(figsize=(8, 6))
            # Plot baseline first
            plt.plot(
                plot_thresholds,
                plot_shots_needed_base * self.shot_factor,
                marker="o",
                linewidth=2,
                label="VQE",
            )
            # Plot main approach
            plt.plot(
                plot_thresholds,
                plot_shots_needed_main * self.shot_factor,
                marker="o",
                linewidth=2,
                label="Tree VQA",
            )

        if len(last_valid_idx) > 0:
            last_valid_idx = last_valid_idx[-1]
            last_valid_threshold = thresholds[last_valid_idx]

            # Calculate shot savings at the last valid point
            main_shots = shots_needed_main[last_valid_idx] * self.shot_factor
            base_shots = shots_needed_base[last_valid_idx] * self.shot_factor
            shot_savings = base_shots - main_shots
            savings_percentage = (
                (shot_savings / base_shots) * 100 if base_shots > 0 else 0
            )
            saving_ratio = base_shots / main_shots

            # Draw vertical line at the last valid fidelity threshold for VQE
            plt.axvline(
                x=last_valid_threshold,
                color="red",
                linestyle="--",
                label=f"Max VQE Fidelity: {last_valid_threshold:.3f}\n$\\mathbf{{Shot\ savings:\ {saving_ratio:.1f}x}}$",
            )

            print(f"At fidelity {last_valid_threshold:.3f}:")
            print(f"  Tree VQA shots: {main_shots:.0f}")
            print(f"  VQE shots: {base_shots:.0f}")
            print(f"  Shot savings: {saving_ratio:.1f}")

        # Formatting
        plt.title(
            "Shots Required vs. All Fidelity Thresholds",
            fontsize=16,
        )
        plt.xlabel("Fidelity Threshold", fontsize=14)
        plt.ylabel(
            "Shots Required",
            fontsize=14,
        )
        plt.tick_params(axis="both", which="major", labelsize=12)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(
            f"{self.save_dir}/{self.molecule_name}shots_vs_fidelity_threshold.pdf"
        )
        print(f"{self.save_dir}/{self.molecule_name}shots_vs_fidelity_threshold.pdf")

    def visualize_shots_for_various_fidelity_avg(self, thresholds: list[float]) -> None:
        """
        For each fidelity threshold in 'thresholds':
          - Determine how many shots the main approach and baseline each need
            so that the *average* fidelity across molecules exceeds the threshold.
          - If it's never reached, store `np.nan`.

        Then plot:
          x-axis = fidelity threshold
          y-axis = shots needed based on average fidelity
          Two lines: main approach vs. baseline

        The resulting plot is suitable for a scientific paper with markers,
        thicker lines, grid, and larger fonts.
        """
        # Sort thresholds for left-to-right plotting
        thresholds = sorted(thresholds)

        # Arrays to hold shots needed for each threshold
        shots_needed_main = []
        shots_needed_base = []

        # Pre-compute average fidelity for each checkpoint
        # Main approach
        avg_fidelity_main = []
        for i_ckpt in range(len(self.checkpoint_indices)):
            f = self.compute_fidelity(
                self.optimized_intermediated_data[i_ckpt, :], self.reference_energies
            )
            avg_fidelity_main.append(np.mean(f))
        avg_fidelity_main = np.array(avg_fidelity_main)

        # Baseline
        avg_fidelity_base = []
        for i_ckpt in range(len(self.seperate_vqe_checkpoint_indices)):
            f = self.compute_fidelity(
                self.seperate_vqe_intermediate_data[i_ckpt, :], self.reference_energies
            )
            avg_fidelity_base.append(np.mean(f))
        avg_fidelity_base = np.array(avg_fidelity_base)

        for T in thresholds:
            # ---- MAIN APPROACH ----
            # Find earliest checkpoint where average fidelity exceeds T
            found_ckpt_main = np.nan
            for i_ckpt in range(len(self.checkpoint_indices)):
                if avg_fidelity_main[i_ckpt] >= T:
                    found_ckpt_main = self.checkpoint_indices[i_ckpt]
                    break
            shots_needed_main.append(found_ckpt_main)

            # ---- BASELINE ----
            found_ckpt_base = np.nan
            for i_ckpt in range(len(self.seperate_vqe_checkpoint_indices)):
                if avg_fidelity_base[i_ckpt] >= T:
                    found_ckpt_base = self.seperate_vqe_checkpoint_indices[i_ckpt]
                    break
            shots_needed_base.append(found_ckpt_base * self.num_molecules)

        # Convert to arrays for plotting
        shots_needed_main = np.array(shots_needed_main)
        shots_needed_base = np.array(shots_needed_base)

        with plt.style.context(("science", "grid")):
            plt.figure(figsize=(8, 6))
            # Plot baseline first
            plt.plot(
                thresholds,
                shots_needed_base * self.shot_factor,
                marker="o",
                linewidth=2,
                label="VQE",
            )
            # Plot main approach
            plt.plot(
                thresholds,
                shots_needed_main * self.shot_factor,
                marker="o",
                linewidth=2,
                label="Tree VQA",
            )

            # Find the last valid point for VQE baseline
            last_valid_idx = np.where(~np.isnan(shots_needed_base))[0]
            if len(last_valid_idx) > 0:
                last_valid_idx = last_valid_idx[-1]
                last_valid_threshold = thresholds[last_valid_idx]

                # Calculate shot savings at the last valid point
                main_shots = shots_needed_main[last_valid_idx] * self.shot_factor
                base_shots = shots_needed_base[last_valid_idx] * self.shot_factor
                shot_savings = base_shots - main_shots
                savings_percentage = (
                    (shot_savings / base_shots) * 100 if base_shots > 0 else 0
                )
                saving_ratio = base_shots / main_shots
                # Draw vertical line at the last valid fidelity threshold for VQE
                plt.axvline(
                    x=last_valid_threshold,
                    color="red",
                    linestyle="--",
                    label=f"Max VQE Fidelity: {last_valid_threshold:.3f}\nShot savings: {saving_ratio:.1f}x",
                )

                print(f"At average fidelity {last_valid_threshold:.3f}:")
                print(f"  Tree VQA shots: {main_shots:.0f}")
                print(f"  VQE shots: {base_shots:.0f}")
                print(f"  Shot savings: {shot_savings:.0f} ({savings_percentage:.1f}%)")

            # Formatting
            plt.title("Shots Required vs. Average Fidelity Threshold", fontsize=16)
            plt.xlabel("Average Fidelity Threshold", fontsize=14)
            plt.ylabel(
                "Shots Required",
                fontsize=14,
            )
            plt.grid(True, alpha=0.3)
            plt.tick_params(axis="both", which="major", labelsize=12)
            plt.legend(fontsize=12)
            plt.tight_layout()
            plt.savefig(
                f"{self.save_dir}/{self.molecule_name}shots_vs_avg_fidelity_threshold.pdf"
            )
            print(
                f"{self.save_dir}/{self.molecule_name}shots_vs_avg_fidelity_threshold.pdf"
            )

    def visualize_shot_reduction_avg_fidelity(self, fixed_fidelity: float) -> None:
        """
        1(a). Compare average fidelity vs. shots for main approach vs. baseline.
        Identify earliest checkpoint at which average fidelity >= fixed_fidelity.
        """
        num_checkpoints_main = len(self.checkpoint_indices)
        num_checkpoints_base = len(self.seperate_vqe_checkpoint_indices)

        # Compute average fidelity over all molecules, main approach
        avg_fidelities_main = []
        for idx in range(num_checkpoints_main):
            fidelity_ckpt = self.compute_fidelity(
                self.optimized_intermediated_data[idx, :], self.reference_energies
            )
            avg_fidelities_main.append(np.mean(fidelity_ckpt))

        # Compute average fidelity for baseline
        avg_fidelities_base = []
        for idx in range(num_checkpoints_base):
            fidelity_ckpt_base = self.compute_fidelity(
                self.seperate_vqe_intermediate_data[idx, :], self.reference_energies
            )
            avg_fidelities_base.append(np.mean(fidelity_ckpt_base))

        avg_fidelities_main = np.array(avg_fidelities_main)
        avg_fidelities_base = np.array(avg_fidelities_base)

        # Find earliest checkpoint for main approach
        passing_main = np.where(avg_fidelities_main >= fixed_fidelity)[0]
        if len(passing_main) > 0:
            earliest_idx_main = passing_main[0]
            shots_main = self.checkpoint_indices[earliest_idx_main]
            print(
                f"[Main Approach] Average fidelity >= {fixed_fidelity} at checkpoint {earliest_idx_main} "
                f"(Shots = {shots_main})."
            )
        else:
            print(f"[Main Approach] Average fidelity never reaches {fixed_fidelity}.")

        # Find earliest checkpoint for baseline
        passing_base = np.where(avg_fidelities_base >= fixed_fidelity)[0]
        if len(passing_base) > 0:
            earliest_idx_base = passing_base[0]
            shots_base = self.seperate_vqe_checkpoint_indices[earliest_idx_base]
            print(
                f"[Baseline] Average fidelity >= {fixed_fidelity} at checkpoint {earliest_idx_base} "
                f"(Shots = {shots_base})."
            )
        else:
            print(f"[Baseline] Average fidelity never reaches {fixed_fidelity}.")

        # Plot average fidelity vs shots
        plt.figure(figsize=(8, 6))
        plt.plot(
            self.seperate_vqe_checkpoint_indices * self.shot_factor,
            avg_fidelities_base,
            marker="o",
            label="Baseline (Separate VQE)",
        )
        plt.plot(
            self.checkpoint_indices * self.shot_factor,
            avg_fidelities_main,
            marker="o",
            label="Tree VQA",
        )
        plt.axhline(y=fixed_fidelity, linestyle="--")
        plt.xlabel("Shots")
        plt.ylabel("Average Fidelity")
        plt.title(
            f"Shot Reduction: Average Fidelity vs Shots (Target = {fixed_fidelity})"
        )
        plt.ylim([0, 1.05])
        plt.legend()
        plt.show()

    def visualize_fidelity_for_fixed_shot(self, shot_budget: float) -> None:
        """
        2. Compare fidelity distribution under a fixed shot budget for:
           - Main approach
           - Baseline.

        We'll pick the largest checkpoint <= shot_budget for each approach,
        then plot histograms of the fidelity across all molecules and print summary stats.
        """
        # Main approach: find checkpoint within shot_budget
        idx_candidates_main = np.where(self.checkpoint_indices <= shot_budget)[0]
        if len(idx_candidates_main) == 0:
            print(
                "[Main Approach] No checkpoints are within the requested shot budget."
            )
            fidelity_main = None
        else:
            chosen_idx_main = idx_candidates_main[-1]
            fidelity_main = self.compute_fidelity(
                self.optimized_intermediated_data[chosen_idx_main, :],
                self.reference_energies,
            )
            print(
                f"[Main Approach] Using checkpoint {chosen_idx_main} with shots "
                f"{self.checkpoint_indices[chosen_idx_main]} (<= {shot_budget})."
            )
            print(
                f"  Avg fidelity: {np.mean(fidelity_main):.3f}\n"
                f"  Min fidelity: {np.min(fidelity_main):.3f}\n"
                f"  Max fidelity: {np.max(fidelity_main):.3f}"
            )
        baseline_shot_budget = shot_budget / self.num_molecules
        # Baseline: find checkpoint within shot_budget
        idx_candidates_base = np.where(
            self.seperate_vqe_checkpoint_indices <= baseline_shot_budget
        )[0]
        if len(idx_candidates_base) == 0:
            print("[Baseline] No checkpoints are within the requested shot budget.")
            fidelity_base = None
        else:
            chosen_idx_base = idx_candidates_base[-1]
            fidelity_base = self.compute_fidelity(
                self.seperate_vqe_intermediate_data[chosen_idx_base, :],
                self.reference_energies,
            )
            print(
                f"[Baseline] Using checkpoint {chosen_idx_base} with shots "
                f"{self.seperate_vqe_checkpoint_indices[chosen_idx_base]} (<= {shot_budget})."
            )
            print(
                f"  Avg fidelity: {np.mean(fidelity_base):.3f}\n"
                f"  Min fidelity: {np.min(fidelity_base):.3f}\n"
                f"  Max fidelity: {np.max(fidelity_base):.3f}"
            )

        # Optional: plot distribution of fidelities for both approaches (if found)
        if fidelity_main is not None or fidelity_base is not None:
            opt_avg, vqe_avg = np.mean(fidelity_main), np.mean(fidelity_base)
            gain = opt_avg - vqe_avg

            # Plot
            plt.figure(figsize=(8, 6))
            bars = plt.bar(
                ["Optimized", "Separate VQE"],
                [opt_avg, vqe_avg],
                color=["#2ca02c", "#1f77b4"],
            )
            plt.ylabel("Average Fidelity", fontsize=12)
            plt.title(
                f"Fidelity Comparison @ {shot_budget} Shots\n"
                f"Gain: {gain:.3f} ({gain / vqe_avg:.1%})",
                fontsize=14,
            )
            plt.ylim(0, 1.05)
            for bar in bars:
                h = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    h,
                    f"{h:.3f}",
                    ha="center",
                    va="bottom",
                )
            plt.grid(axis="y", alpha=0.4)
            plt.savefig("fidelity_comparison.pdf")

    def visualize_fidelity_for_various_shot_budgets(
        self, shot_budgets: list[float]
    ) -> None:
        """
        Demonstrate how fidelity (avg, min, max) changes as we vary the shot budget
        for both the main approach and the baseline.

        We produce a single line plot of average fidelity vs. shot_budget,
        with fill between min and max to show the spread across all molecules.
        """
        # Sort budgets so our plot lines move left-to-right
        shot_budgets = sorted(shot_budgets)

        avg_main, min_main, max_main = [], [], []
        avg_base, min_base, max_base = [], [], []
        improvements = []  # Track improvements

        for sb in shot_budgets:
            # ---- Main Approach ----
            idx_candidates_main = np.where(self.checkpoint_indices <= sb)[0]
            if len(idx_candidates_main) == 0:
                avg_main.append(np.nan)
                min_main.append(np.nan)
                max_main.append(np.nan)
                improvements.append(np.nan)
            else:
                chosen_idx_main = idx_candidates_main[-1]
                f_main = self.compute_fidelity(
                    self.optimized_intermediated_data[chosen_idx_main, :],
                    self.reference_energies,
                )
                avg_main.append(np.mean(f_main))
                min_main.append(np.min(f_main))
                max_main.append(np.max(f_main))

            # ---- Baseline ----
            seperate_vqe_shot_budget = sb / self.num_molecules
            idx_candidates_base = np.where(
                self.seperate_vqe_checkpoint_indices <= seperate_vqe_shot_budget
            )[0]
            if len(idx_candidates_base) == 0:
                avg_base.append(np.nan)
                min_base.append(np.nan)
                max_base.append(np.nan)
            else:
                chosen_idx_base = idx_candidates_base[-1]
                f_base = self.compute_fidelity(
                    self.seperate_vqe_intermediate_data[chosen_idx_base, :],
                    self.reference_energies,
                )
                avg_base.append(np.mean(f_base))
                min_base.append(np.min(f_base))
                max_base.append(np.max(f_base))

        # Convert to arrays
        avg_main = np.array(avg_main)
        min_main = np.array(min_main)
        max_main = np.array(max_main)
        avg_base = np.array(avg_base)
        min_base = np.array(min_base)
        max_base = np.array(max_base)

        # ---- Create a nice figure for the data ----
        with plt.style.context(("science", "grid")):
            plt.figure(figsize=(8, 6))
            shot_budgets = np.array(shot_budgets)
            # Plot baseline average fidelity
            plt.plot(
                shot_budgets,
                avg_base,
                label="Separate VQE",
                linewidth=2,
            )
            plt.fill_between(shot_budgets, min_base, max_base, alpha=0.15)
            # Plot main approach average fidelity
            plt.plot(
                shot_budgets,
                avg_main,
                label="Tree VQA",
                linewidth=2,
            )
            # Shaded region between min and max
            plt.fill_between(shot_budgets, min_main, max_main, alpha=0.15)

            # Labels, title, etc.
            plt.xlabel("Shot Budget", fontsize=14)
            plt.ylabel("Fidelity (Average)", fontsize=14)
            plt.title(
                "Fidelity vs. Shot Budget",
                fontsize=16,
            )
            STRIDE = shot_budgets.size // 6
            # Enable grid
            plt.grid(True, alpha=0.3)
            # Make tick labels a bit bigger
            plt.tick_params(axis="both", which="major", labelsize=12)
            plt.xticks(
                ticks=shot_budgets[::STRIDE],
                labels=[
                    f"{sb * self.shot_factor:.3g}" for sb in shot_budgets[::STRIDE]
                ],
            )

            plt.legend(fontsize=12)
            # plt.ylim([0.98, 1.00])
            plt.tight_layout()
            plt.savefig(
                f"{self.save_dir}/{self.molecule_name}fidelity_vs_shot_budget.pdf"
            )

    def get_shot_savings(
        self, threshold: float = None, is_avg: bool = False
    ) -> tuple[float, float]:
        """
        Calculate the shot savings and savings percentage at a given fidelity threshold.
        If threshold is not specified, it will use the highest fidelity threshold
        that both the Tree VQA and baseline VQE methods can achieve.

        Parameters
        ----------
        threshold : float, optional
            The fidelity threshold to calculate savings at. If None, uses the highest
            achievable threshold by both methods.
        is_avg : bool, optional
            If True, calculate savings when the average fidelity meets the threshold.
            If False (default), calculate when all fidelity values meet the threshold.

        Returns
        -------
        tuple[float, float]
            A tuple containing (shot_savings, savings_percentage)

        Notes
        -----
        Shot savings is calculated as baseline_shots - main_shots
        Savings percentage is (shot_savings / baseline_shots) * 100
        """
        # Pre-compute fidelity for each checkpoint
        fidelity_main = []
        for i_ckpt in range(len(self.checkpoint_indices)):
            f = self.compute_fidelity(
                self.optimized_intermediated_data[i_ckpt, :], self.reference_energies
            )
            fidelity_main.append(f)
        fidelity_main = np.array(fidelity_main)

        fidelity_base = []
        for i_ckpt in range(len(self.seperate_vqe_checkpoint_indices)):
            f = self.compute_fidelity(
                self.seperate_vqe_intermediate_data[i_ckpt, :], self.reference_energies
            )
            fidelity_base.append(f)
        fidelity_base = np.array(fidelity_base)

        # If threshold not provided, find the highest threshold that both methods can achieve
        if threshold is None:
            if is_avg:
                # Find highest average fidelity thresholds that each method can achieve
                max_threshold_main = max(
                    np.mean(fidelity_level)
                    for fidelity_level in fidelity_main
                    if len(fidelity_level) > 0
                )
                max_threshold_base = max(
                    np.mean(fidelity_level)
                    for fidelity_level in fidelity_base
                    if len(fidelity_level) > 0
                )
            else:
                # Find highest fidelity thresholds that each method can achieve
                max_threshold_main = max(
                    np.min(fidelity_level)
                    for fidelity_level in fidelity_main
                    if len(fidelity_level) > 0
                )
                max_threshold_base = max(
                    np.min(fidelity_level)
                    for fidelity_level in fidelity_base
                    if len(fidelity_level) > 0
                )

            # Use the lower of the two as our threshold
            threshold = min(max_threshold_main, max_threshold_base)
            print(f"Using automatically determined threshold: {threshold:.3f}")

        # Find shots needed for main approach at this threshold
        found_ckpt_main = np.nan
        for i_ckpt in range(len(self.checkpoint_indices)):
            if is_avg:
                if np.mean(fidelity_main[i_ckpt, :]) >= threshold:
                    found_ckpt_main = self.checkpoint_indices[i_ckpt]
                    break
            else:
                if np.all(fidelity_main[i_ckpt, :] >= threshold):
                    found_ckpt_main = self.checkpoint_indices[i_ckpt]
                    break

        # Find shots needed for baseline at this threshold
        found_ckpt_base = np.nan
        for i_ckpt in range(len(self.seperate_vqe_checkpoint_indices)):
            if is_avg:
                if np.mean(fidelity_base[i_ckpt, :]) >= threshold:
                    found_ckpt_base = self.seperate_vqe_checkpoint_indices[i_ckpt]
                    break
            else:
                if np.all(fidelity_base[i_ckpt, :] >= threshold):
                    found_ckpt_base = self.seperate_vqe_checkpoint_indices[i_ckpt]
                    break

        # Calculate shot counts with the shot factor
        main_shots = found_ckpt_main * self.shot_factor
        base_shots = found_ckpt_base * self.num_molecules * self.shot_factor

        # Calculate savings
        shot_savings = base_shots - main_shots
        savings_percentage = (shot_savings / base_shots) * 100 if base_shots > 0 else 0

        fidelity_type = "average fidelity" if is_avg else "all fidelities"
        print(f"At {fidelity_type} {threshold:.3f}:")
        print(f"  Tree VQA shots: {main_shots:.0f}")
        print(f"  VQE shots: {base_shots:.0f}")
        print(f"  Shot savings: {shot_savings:.0f} ({savings_percentage:.1f}%)")

        return shot_savings, savings_percentage

    def plot_fidelity_vs_shot_budget_subplot(
        self,
        ax: plt.Axes,
        shot_budgets: list[float],
    ) -> None:
        """
        Plot fidelity vs shot budget on a given subplot.

        Parameters
        ----------
        ax : plt.Axes
            The subplot to plot on
        shot_budgets : list[float]
            List of shot budgets to evaluate
        """
        avg_main, min_main, max_main = [], [], []
        avg_base, min_base, max_base = [], [], []

        for sb in shot_budgets:
            # ---- Main Approach ----
            idx_candidates_main = np.where(self.checkpoint_indices <= sb)[0]
            if len(idx_candidates_main) == 0:
                avg_main.append(np.nan)
                min_main.append(np.nan)
                max_main.append(np.nan)
            else:
                chosen_idx_main = idx_candidates_main[-1]
                f_main = self.compute_fidelity(
                    self.optimized_intermediated_data[chosen_idx_main, :],
                    self.reference_energies,
                )
                avg_main.append(np.mean(f_main))
                min_main.append(np.min(f_main))
                max_main.append(np.max(f_main))

            # ---- Baseline ----
            seperate_vqe_shot_budget = sb / self.num_molecules
            idx_candidates_base = np.where(
                self.seperate_vqe_checkpoint_indices <= seperate_vqe_shot_budget
            )[0]
            if len(idx_candidates_base) == 0:
                avg_base.append(np.nan)
                min_base.append(np.nan)
                max_base.append(np.nan)
            else:
                chosen_idx_base = idx_candidates_base[-1]
                f_base = self.compute_fidelity(
                    self.seperate_vqe_intermediate_data[chosen_idx_base, :],
                    self.reference_energies,
                )
                avg_base.append(np.mean(f_base))
                min_base.append(np.min(f_base))
                max_base.append(np.max(f_base))

        # Convert to arrays
        avg_main = np.array(avg_main)
        min_main = np.array(min_main)
        max_main = np.array(max_main)
        avg_base = np.array(avg_base)
        min_base = np.array(min_base)
        max_base = np.array(max_base)

        shot_budgets = np.array(shot_budgets)
        # Plot baseline average fidelity
        ax.plot(
            shot_budgets * self.shot_factor,
            avg_base,
            label="Separate VQE",
            linewidth=3,
        )
        ax.fill_between(shot_budgets * self.shot_factor, min_base, max_base, alpha=0.15)
        # Plot main approach average fidelity
        ax.plot(
            shot_budgets * self.shot_factor,
            avg_main,
            label="Tree VQA",
            linewidth=3,
        )
        # Shaded region between min and max
        ax.fill_between(shot_budgets * self.shot_factor, min_main, max_main, alpha=0.20)
        ax.set_title(f"${self.molecule_name}$", fontsize=26)
        ax.tick_params(axis="both", which="major", labelsize=26)

        # Format x-axis with scientific notation
        ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
        # Adjust the offset text size and position
        offset = ax.get_xaxis().get_offset_text()
        offset.set_fontsize(22)
        offset.set_position((1.1, -0.2))  # Adjust position of scientific notation

    @staticmethod
    def plot_combined_fidelity_vs_shot_budget(
        analyses: list["Result_analysis"],
        shot_budgets_list: list[list[float]],
        save_dir: str = "exp-fig",
    ) -> None:
        """
        Create a combined plot with subplots for all molecules.

        Parameters
        ----------
        analyses : list[Result_analysis]
            List of Result_analysis objects for each molecule
        shot_budgets_list : list[list[float]]
            List of shot budgets for each molecule
        save_dir : str
            Directory to save the combined plot
        """
        with plt.style.context(("science", "grid")):
            fig, axes = plt.subplots(2, 3, figsize=(30, 12))
            axes = axes.flatten()

            for ax, analysis, shot_budgets in zip(axes, analyses, shot_budgets_list):
                analysis.plot_fidelity_vs_shot_budget_subplot(ax, shot_budgets)

            # Add common labels
            fig.supxlabel(
                "Shot Budget", fontsize=28, y=0.02
            )  # Adjust y position to avoid overlap
            fig.supylabel("Fidelity ", fontsize=28)

            # Add legend to all subplots
            for ax in axes:
                ax.legend(fontsize=24)

            plt.tight_layout()
            plt.savefig(
                f"{save_dir}/combined_fidelity_vs_shot_budget.pdf", bbox_inches="tight"
            )
            print(
                f"Saved combined plot to {save_dir}/combined_fidelity_vs_shot_budget.pdf"
            )

    def plot_shots_vs_fidelity_subplot(
        self,
        ax: plt.Axes,
        thresholds: list[float],
    ) -> None:
        """
        Plot shots needed vs fidelity threshold on a given subplot.

        Parameters
        ----------
        ax : plt.Axes
            The subplot to plot on
        thresholds : list[float]
            List of fidelity thresholds to evaluate
        """
        # Arrays to hold shots needed for each threshold
        shots_needed_main = []
        shots_needed_base = []

        # Pre-compute fidelity for each checkpoint
        # Main approach shape: (num_checkpoints, num_molecules)
        fidelity_main = []
        for i_ckpt in range(len(self.checkpoint_indices)):
            f = self.compute_fidelity(
                self.optimized_intermediated_data[i_ckpt, :], self.reference_energies
            )
            fidelity_main.append(f)
        fidelity_main = np.array(fidelity_main)

        # Baseline shape: (num_checkpoints, num_molecules)
        fidelity_base = []
        for i_ckpt in range(len(self.seperate_vqe_checkpoint_indices)):
            f = self.compute_fidelity(
                self.seperate_vqe_intermediate_data[i_ckpt, :], self.reference_energies
            )
            fidelity_base.append(f)
        fidelity_base = np.array(fidelity_base)

        for T in thresholds:
            # ---- MAIN APPROACH ----
            found_ckpt_main = np.nan
            for i_ckpt in range(len(self.checkpoint_indices)):
                if np.all(fidelity_main[i_ckpt, :] >= T):
                    found_ckpt_main = self.checkpoint_indices[i_ckpt]
                    break
            shots_needed_main.append(found_ckpt_main)

            # ---- BASELINE ----
            found_ckpt_base = np.nan
            for i_ckpt in range(len(self.seperate_vqe_checkpoint_indices)):
                if np.all(fidelity_base[i_ckpt, :] >= T):
                    found_ckpt_base = self.seperate_vqe_checkpoint_indices[i_ckpt]
                    break
            shots_needed_base.append(found_ckpt_base * self.num_molecules)

        # Convert to arrays for plotting
        shots_needed_main = np.array(shots_needed_main)
        shots_needed_base = np.array(shots_needed_base)

        # Plot baseline first
        ax.plot(
            thresholds,
            shots_needed_base * self.shot_factor,
            marker="o",
            linewidth=3,
            label="Separate VQE",
        )
        # Plot main approach
        ax.plot(
            thresholds,
            shots_needed_main * self.shot_factor,
            marker="o",
            linewidth=3,
            label="Tree VQA",
        )

        # Find the last valid point for VQE baseline
        last_valid_idx = np.where(~np.isnan(shots_needed_base))[0]
        if len(last_valid_idx) > 0:
            last_valid_idx = last_valid_idx[-1]
            last_valid_threshold = thresholds[last_valid_idx]

            # Calculate shot savings at the last valid point
            main_shots = shots_needed_main[last_valid_idx] * self.shot_factor
            base_shots = shots_needed_base[last_valid_idx] * self.shot_factor
            saving_ratio = base_shots / main_shots

            # Draw vertical line at the last valid fidelity threshold for VQE
            ax.axvline(
                x=last_valid_threshold,
                color="red",
                linestyle="--",
                label=f"Max VQE Fidelity: {last_valid_threshold:.3f}\n$\\mathbf{{Shot\ savings:\ {saving_ratio:.1f}x}}$",
            )

        ax.set_title(f"${self.molecule_name}$", fontsize=26)
        ax.tick_params(axis="both", which="major", labelsize=26)

        # Format y-axis with scientific notation
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        # Adjust the offset text size and position
        offset = ax.get_yaxis().get_offset_text()
        offset.set_fontsize(22)
        # Move the scientific notation to top of y-axis
        offset.set_position((-0.1, 1.05))

    @staticmethod
    def plot_combined_shots_vs_fidelity(
        analyses: list["Result_analysis"],
        fidelity_thresholds_list: list[list[float]],
        save_dir: str = "exp-fig",
    ) -> None:
        """
        Create a combined plot with subplots for all molecules showing shots needed vs fidelity threshold.

        Parameters
        ----------
        analyses : list[Result_analysis]
            List of Result_analysis objects for each molecule
        fidelity_thresholds_list : list[list[float]]
            List of fidelity thresholds for each molecule
        save_dir : str
            Directory to save the combined plot
        """
        with plt.style.context(("science", "grid")):
            # Match figure size with combined_fidelity_vs_shot_budget
            fig, axes = plt.subplots(2, 3, figsize=(30, 12))
            axes = axes.flatten()

            for ax, analysis, thresholds in zip(
                axes, analyses, fidelity_thresholds_list
            ):
                analysis.plot_shots_vs_fidelity_subplot(ax, thresholds)

            # Add common labels with adjusted position
            fig.supxlabel("Fidelity Threshold", fontsize=28, y=0.02)
            fig.supylabel("Shots Required", fontsize=28, x=0.02)

            # Add legend to all subplots
            for ax in axes:
                ax.legend(fontsize=24)

            # Adjust spacing between subplots and margins
            plt.subplots_adjust(
                left=0.08, bottom=0.08, right=0.94, top=0.97, hspace=0.13, wspace=0.12
            )

            plt.savefig(
                f"{save_dir}/combined_shots_vs_fidelity.pdf",
                bbox_inches=None,  # Don't use tight_layout to preserve our manual margins
                dpi=300,
            )
            print(f"Saved combined plot to {save_dir}/combined_shots_vs_fidelity.pdf")

    def analyze_shot_gain_without_reference(
        self, shot_budget: float, use_relative_improvement: bool = True
    ) -> dict:
        """
        Analyze shot gain across molecules without knowing reference energy.
        This method compares the energy improvements achieved by both methods
        at the same shot budget.

        Parameters
        ----------
        shot_budget : float
            The shot budget to compare both methods at
        use_relative_improvement : bool, optional
            If True, calculate relative improvement from baseline energy.
            If False, use absolute energy differences.

        Returns
        -------
        dict
            Dictionary containing analysis results with keys:
            - 'main_energies': energies from main approach
            - 'baseline_energies': energies from baseline approach
            - 'improvements': improvement metrics for each molecule
            - 'avg_improvement': average improvement across molecules
            - 'improvement_ratio': ratio of average improvements
            - 'molecules_better': number of molecules where main > baseline
            - 'shot_budget_used': actual shot budget used
        """
        # Find appropriate checkpoints for main approach
        idx_candidates_main = np.where(self.checkpoint_indices <= shot_budget)[0]
        if len(idx_candidates_main) == 0:
            raise ValueError("[Main Approach] No checkpoints within shot budget")

        chosen_idx_main = idx_candidates_main[-1]
        main_energies = self.optimized_intermediated_data[chosen_idx_main, :]
        actual_shots_main = self.checkpoint_indices[chosen_idx_main]

        # Find appropriate checkpoints for baseline (per-molecule budget)
        baseline_shot_budget = shot_budget / self.num_molecules
        idx_candidates_base = np.where(
            self.seperate_vqe_checkpoint_indices <= baseline_shot_budget
        )[0]
        if len(idx_candidates_base) == 0:
            raise ValueError("[Baseline] No checkpoints within shot budget")

        chosen_idx_base = idx_candidates_base[-1]
        baseline_energies = self.seperate_vqe_intermediate_data[chosen_idx_base, :]
        actual_shots_base = self.seperate_vqe_checkpoint_indices[chosen_idx_base]

        # Calculate improvements (lower energy is better for VQE)
        # Improvement = how much lower the energy is compared to baseline
        energy_differences = baseline_energies - main_energies

        if use_relative_improvement:
            # Relative improvement as percentage of baseline energy magnitude
            improvements = (energy_differences / np.abs(baseline_energies)) * 100
        else:
            # Absolute energy differences
            improvements = energy_differences

        # Calculate statistics
        avg_improvement_main = np.mean(improvements)
        avg_improvement_baseline = 0.0  # baseline is reference point
        molecules_better = np.sum(
            energy_differences > 0
        )  # main method achieves lower energy

        results = {
            "main_energies": main_energies,
            "baseline_energies": baseline_energies,
            "energy_differences": energy_differences,
            "improvements": improvements,
            "avg_improvement": avg_improvement_main,
            "improvement_ratio": np.inf
            if avg_improvement_baseline == 0
            else avg_improvement_main / avg_improvement_baseline,
            "molecules_better": molecules_better,
            "total_molecules": self.num_molecules,
            "fraction_better": molecules_better / self.num_molecules,
            "shot_budget_used": shot_budget,
            "actual_shots_main": actual_shots_main,
            "actual_shots_baseline": actual_shots_base,
            "use_relative": use_relative_improvement,
        }

        return results

    def visualize_shot_gain_without_reference(
        self, shot_budget: float, use_relative_improvement: bool = True
    ) -> None:
        """
        Visualize shot gain analysis across molecules without reference energy.
        Creates multiple plots showing the comparison between methods.

        Parameters
        ----------
        shot_budget : float
            The shot budget to compare both methods at
        use_relative_improvement : bool, optional
            If True, show relative improvement percentages.
            If False, show absolute energy differences.
        """
        results = self.analyze_shot_gain_without_reference(
            shot_budget, use_relative_improvement
        )

        # Print summary statistics
        print(f"\n=== Shot Gain Analysis at {shot_budget} shots ===")
        print(f"Main approach actual shots: {results['actual_shots_main']}")
        print(f"Baseline actual shots per molecule: {results['actual_shots_baseline']}")
        print(
            f"Molecules where main method is better: {results['molecules_better']}/{results['total_molecules']} ({results['fraction_better']:.1%})"
        )

        if use_relative_improvement:
            print(f"Average relative improvement: {results['avg_improvement']:.2f}%")
        else:
            print(f"Average absolute improvement: {results['avg_improvement']:.6f}")

        with plt.style.context(("science", "grid")):
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

            # Plot 1: Energy comparison across molecules
            x_pos = np.arange(len(results["main_energies"]))
            width = 0.35

            ax1.bar(
                x_pos - width / 2,
                results["baseline_energies"],
                width,
                label="Separate VQE",
                alpha=0.8,
                color="#1f77b4",
            )
            ax1.bar(
                x_pos + width / 2,
                results["main_energies"],
                width,
                label="Tree VQA",
                alpha=0.8,
                color="#ff7f0e",
            )

            ax1.set_xlabel("Hamiltonian Index", fontsize=12)
            ax1.set_ylabel("Energy", fontsize=12)
            ax1.set_title("Energy Comparison Across Hamiltonians", fontsize=14)
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot 2: Improvement distribution
            if use_relative_improvement:
                ax2.hist(
                    results["improvements"],
                    bins=min(20, len(results["improvements"]) // 2),
                    alpha=0.7,
                    edgecolor="black",
                )
                ax2.set_xlabel("Relative Improvement (\%)", fontsize=12)
                ax2.set_title("Distribution of Relative Improvements", fontsize=14)
                ax2.axvline(
                    results["avg_improvement"],
                    color="red",
                    linestyle="--",
                    label=f"Mean: {results['avg_improvement']:.2f}%",
                )
            else:
                ax2.hist(
                    results["improvements"],
                    bins=min(20, len(results["improvements"]) // 2),
                    alpha=0.7,
                    edgecolor="black",
                )
                ax2.set_xlabel("Absolute Energy Improvement", fontsize=12)
                ax2.set_title("Distribution of Absolute Improvements", fontsize=14)
                ax2.axvline(
                    results["avg_improvement"],
                    color="red",
                    linestyle="--",
                    label=f"Mean: {results['avg_improvement']:.6f}",
                )

            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Plot 3: Molecule-by-molecule improvement
            colors = ["green" if imp > 0 else "red" for imp in results["improvements"]]
            ax3.bar(x_pos, results["improvements"], color=colors, alpha=0.7)
            ax3.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
            ax3.set_xlabel("Hamiltonian Index", fontsize=12)

            if use_relative_improvement:
                ax3.set_ylabel("Relative Improvement (\%)", fontsize=12)
                ax3.set_title("Per-Hamiltonian Relative Improvement", fontsize=14)
            else:
                ax3.set_ylabel("Absolute Energy Improvement", fontsize=12)
                ax3.set_title("Per-Hamiltonian Absolute Improvement", fontsize=14)

            ax3.grid(True, alpha=0.3)

            # Plot 4: Summary statistics
            ax4.axis("off")

            # Create summary text
            summary_text = f"""
Shot Budget Analysis Summary

Total Shot Budget: {shot_budget:,}
Main Method Shots Used: {results["actual_shots_main"]:,}
Baseline Shots Used (per Hamiltonian): {results["actual_shots_baseline"]:,}

Performance Metrics:
• Hamiltonians Improved: {results["molecules_better"]}/{results["total_molecules"]} ({results["fraction_better"]:.1%})
• Best Improvement: {np.max(results["improvements"]):.3f}{"%" if use_relative_improvement else ""}
• Worst Result: {np.min(results["improvements"]):.3f}{"%" if use_relative_improvement else ""}
• Average Improvement: {results["avg_improvement"]:.3f}{"%" if use_relative_improvement else ""}
• Std Deviation: {np.std(results["improvements"]):.3f}{"%" if use_relative_improvement else ""}
            """

            ax4.text(
                0.05,
                0.95,
                summary_text,
                transform=ax4.transAxes,
                fontsize=11,
                verticalalignment="top",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
            )

            plt.tight_layout()

            # Save the plot
            filename_suffix = "relative" if use_relative_improvement else "absolute"
            save_path = f"{self.save_dir}/{self.molecule_name}shot_gain_analysis_{filename_suffix}.pdf"
            plt.savefig(save_path)

            print(f"\nPlot saved to: {save_path}")
            plt.show()

    def compare_shot_efficiency_across_budgets(
        self, shot_budgets: list[float], use_relative_improvement: bool = True
    ) -> dict:
        """
        Compare shot efficiency across different shot budgets without reference energy.

        Parameters
        ----------
        shot_budgets : list[float]
            List of shot budgets to analyze
        use_relative_improvement : bool, optional
            Whether to use relative or absolute improvements

        Returns
        -------
        dict
            Dictionary with shot budgets as keys and analysis results as values
        """
        shot_budgets = sorted(shot_budgets)
        all_results = {}

        for budget in shot_budgets:
            try:
                results = self.analyze_shot_gain_without_reference(
                    budget, use_relative_improvement
                )
                all_results[budget] = results
            except ValueError as e:
                print(f"Skipping budget {budget}: {e}")
                continue

        return all_results

    def visualize_efficiency_trends(
        self, shot_budgets: list[float], use_relative_improvement: bool = True
    ) -> None:
        """
        Visualize how shot efficiency changes across different shot budgets.

        Parameters
        ----------
        shot_budgets : list[float]
            List of shot budgets to analyze
        use_relative_improvement : bool, optional
            Whether to use relative or absolute improvements
        """
        all_results = self.compare_shot_efficiency_across_budgets(
            shot_budgets, use_relative_improvement
        )

        if not all_results:
            print("No valid results found for any shot budget")
            return

        # Extract data for plotting
        budgets = list(all_results.keys())
        avg_improvements = [all_results[b]["avg_improvement"] for b in budgets]
        fractions_better = [all_results[b]["fraction_better"] for b in budgets]
        efficiency_ratios = [
            (all_results[b]["actual_shots_baseline"] * self.num_molecules)
            / all_results[b]["actual_shots_main"]
            for b in budgets
        ]

        with plt.style.context(("science", "grid")):
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

            # Plot 1: Average improvement vs shot budget
            ax1.plot(budgets, avg_improvements, "o-", linewidth=2, markersize=8)
            ax1.set_xlabel("Shot Budget", fontsize=14)
            ylabel = (
                "Average Relative Improvement (%)"
                if use_relative_improvement
                else "Average Absolute Improvement"
            )
            ax1.set_ylabel(ylabel, fontsize=14)
            ax1.set_title("Average Improvement vs Shot Budget", fontsize=16)
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis="both", which="major", labelsize=12)

            # Plot 2: Fraction of molecules improved vs shot budget
            ax2.plot(
                budgets,
                fractions_better,
                "o-",
                linewidth=2,
                markersize=8,
                color="green",
            )
            ax2.set_xlabel("Shot Budget", fontsize=14)
            ax2.set_ylabel("Fraction of Molecules Improved", fontsize=14)
            ax2.set_title("Success Rate vs Shot Budget", fontsize=16)
            ax2.set_ylim(0, 1.05)
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis="both", which="major", labelsize=12)

            # Plot 3: Shot efficiency ratio vs shot budget
            ax3.plot(
                budgets,
                efficiency_ratios,
                "o-",
                linewidth=2,
                markersize=8,
                color="purple",
            )
            ax3.set_xlabel("Shot Budget", fontsize=14)
            ax3.set_ylabel("Shot Efficiency Ratio", fontsize=14)
            ax3.set_title("Shot Efficiency vs Budget", fontsize=16)
            ax3.grid(True, alpha=0.3)
            ax3.tick_params(axis="both", which="major", labelsize=12)

            plt.tight_layout()

            # Save the plot
            filename_suffix = "relative" if use_relative_improvement else "absolute"
            save_path = f"{self.save_dir}/{self.molecule_name}efficiency_trends_{filename_suffix}.pdf"
            plt.savefig(save_path)

            print(f"\nEfficiency trends plot saved to: {save_path}")

    def analyze_shot_efficiency_at_same_energy(self, shot_budget: float) -> dict:
        """
        Find how many shots baseline needs to achieve the same energies as main method.

        Parameters
        ----------
        shot_budget : float
            Shot budget for the main method

        Returns
        -------
        dict
            Dictionary containing shot comparison results for each molecule
        """
        # Get main method energies at the given shot budget
        idx_candidates_main = np.where(self.checkpoint_indices <= shot_budget)[0]
        if len(idx_candidates_main) == 0:
            raise ValueError("[Main Approach] No checkpoints within shot budget")

        chosen_idx_main = idx_candidates_main[-1]
        main_energies = self.optimized_intermediated_data[chosen_idx_main, :]
        actual_shots_main = self.checkpoint_indices[chosen_idx_main]

        # For each molecule, find the baseline shots needed to achieve similar energy
        results_per_molecule = []
        baseline_shots_needed = []
        energy_differences = []

        for mol_idx in range(self.num_molecules):
            target_energy = main_energies[mol_idx]

            # Find closest baseline energy and corresponding shots
            baseline_energies_mol = self.seperate_vqe_intermediate_data[:, mol_idx]
            energy_diffs = np.abs(baseline_energies_mol - target_energy)
            closest_idx = np.argmin(energy_diffs)

            closest_baseline_energy = baseline_energies_mol[closest_idx]
            baseline_shots = self.seperate_vqe_checkpoint_indices[closest_idx]
            energy_diff = closest_baseline_energy - target_energy

            results_per_molecule.append(
                {
                    "molecule_idx": mol_idx,
                    "target_energy": target_energy,
                    "closest_baseline_energy": closest_baseline_energy,
                    "energy_difference": energy_diff,
                    "baseline_shots_needed": baseline_shots,
                    "main_shots": actual_shots_main,
                }
            )

            baseline_shots_needed.append(baseline_shots)
            energy_differences.append(energy_diff)
        # Calculate total shot requirements
        total_main_shots = actual_shots_main * self.shot_factor
        total_baseline_shots = np.sum(baseline_shots_needed) * self.shot_factor
        total_shot_savings = total_baseline_shots - total_main_shots
        efficiency_ratio = total_baseline_shots / total_main_shots

        summary_results = {
            "shot_budget": shot_budget,
            "main_shots_used": actual_shots_main,
            "per_molecule_results": results_per_molecule,
            "baseline_shots_needed": baseline_shots_needed,
            "energy_differences": energy_differences,
            "total_main_shots": total_main_shots,
            "total_baseline_shots": total_baseline_shots,
            "total_shot_savings": total_shot_savings,
            "efficiency_ratio": efficiency_ratio,
            "avg_energy_difference": np.mean(energy_differences),
            "max_energy_difference": np.max(np.abs(energy_differences)),
        }
        return summary_results

    def visualize_shot_efficiency_at_same_energy(self, shot_budget: float) -> None:
        """
        Visualize shot efficiency comparison when achieving the same energy levels.

        Parameters
        ----------
        shot_budget : float
            Shot budget for the main method
        """
        results = self.analyze_shot_efficiency_at_same_energy(shot_budget)

        # Print summary
        print(f"\n=== Shot Efficiency at Same Energy Levels ===")
        print(f"Main method shot budget: {shot_budget}")
        print(f"Main method actual shots used: {results['main_shots_used']}")
        print(f"Total main method shots: {results['total_main_shots']:,.0f}")
        print(f"Total baseline shots needed: {results['total_baseline_shots']:,.0f}")
        print(f"Total shot savings: {results['total_shot_savings']:,.0f}")
        print(f"Efficiency ratio: {results['efficiency_ratio']:.1f}x")
        print(f"Average energy difference: {results['avg_energy_difference']:.6f}")
        print(f"Max energy difference: {results['max_energy_difference']:.6f}")

        # Print per-Hamiltonian details
        print(f"\n=== Per-Hamiltonian Details ===")
        shot_savings_ratio_per_mol = []
        energy_diff_per_mol = []
        for i, mol_result in enumerate(results["per_molecule_results"]):
            baseline_shots = mol_result["baseline_shots_needed"] * self.shot_factor
            main_shots = mol_result["main_shots"] * self.shot_factor
            shot_savings = baseline_shots - main_shots
            savings_ratio = (
                baseline_shots / (main_shots / self.num_molecules)
                if main_shots > 0
                else float("inf")
            )
            energy_diff = mol_result["energy_difference"]
            shot_savings_ratio_per_mol.append(float(savings_ratio))
            energy_diff_per_mol.append(float(energy_diff))

            print(f"Hamiltonian {i}:")
            print(f"  Energy difference: {energy_diff:.6f}")
            print(f"  Shot savings: {shot_savings:,.0f}")
            print(f"  Shot savings ratio: {savings_ratio:.1f}x")

        print(f"Shot savings ratio per molecule: {shot_savings_ratio_per_mol}")
        print(f"Energy difference per molecule: {energy_diff_per_mol}")

        with plt.style.context(("science", "grid")):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            molecule_indices = range(self.num_molecules)
            baseline_shots = results["baseline_shots_needed"]
            main_shots_per_mol = [results["main_shots_used"]] * self.num_molecules

            # Plot 1: Energy difference as box plot
            energy_diffs = results["energy_differences"]
            ax1.boxplot(
                energy_diffs,
                vert=True,
                patch_artist=True,
                boxprops=dict(facecolor="lightblue"),
            )
            ax1.set_ylabel("Energy Difference (Baseline - TreeVQA)", fontsize=12)
            ax1.set_title("Energy Difference Distribution", fontsize=14)
            ax1.grid(True, alpha=0.3)
            ax1.set_xticks([1])
            ax1.set_xticklabels(["All Hamiltonians"])

            # Plot 2: Shot savings per molecule
            shot_savings_per_mol = [
                (baseline_shots[i] - results["main_shots_used"] / self.num_molecules)
                * self.shot_factor
                for i in range(self.num_molecules)
            ]
            colors = [
                "green" if energy_diff > 0 else "lightblue"
                for energy_diff in energy_diffs
            ]

            ax2.bar(
                molecule_indices,
                shot_savings_per_mol,
                color=colors,
            )
            ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
            ax2.set_xlabel("Hamiltonian Index", fontsize=12)
            ax2.set_ylabel("Shot Savings", fontsize=12)
            ax2.set_title("Shot Savings Per Hamiltonian", fontsize=14)
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save the plot
            save_path = (
                f"{self.save_dir}/{self.molecule_name}shot_efficiency_same_energy.pdf"
            )
            plt.savefig(save_path)

            print(f"\nPlot saved to: {save_path}")
            plt.show()
