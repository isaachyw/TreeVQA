import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots

plt.style.use("science")


def compute_and_plot_gram_matrix(
    X, labels=None, save_name=None, title="Gram Matrix Similarity"
):
    """
    Compute and visualize the Gram matrix of a design matrix with data labels.

    Args:
        X (np.ndarray): n x m design matrix
        labels (List[str]): List of n data labels
        title (str): Title for the plot

    Returns:
        np.ndarray: Computed Gram matrix
    """
    # Compute Gram matrix
    gram = X @ X.T

    # Create plot
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        gram, cmap="viridis", annot=False, cbar_kws={"label": "Similarity"}
    )

    # Add labels if provided
    if labels is not None:
        if len(labels) != X.shape[0]:
            raise ValueError("Length of labels must match number of rows in X")
        ax.set_xticks(np.arange(len(labels)) + 0.5)
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticks(np.arange(len(labels)) + 0.5)
        ax.set_yticklabels(labels, rotation=0)

    plt.title(title)
    plt.xlabel("Bondlength")
    plt.ylabel("Bondlength")
    plt.tight_layout()
    plt.savefig(save_name)

    return gram


def standardized_total_variance(X):
    """Returns total variance after z-score normalization"""
    X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    return np.trace(np.cov(X_std, rowvar=False))


import numpy as np


def normalized_total_variance(X):
    """Returns total variance normalized to [0,1] scale"""
    # Standardize the data
    X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # Compute pairwise squared Euclidean distances
    dists = np.sqrt(np.sum((X_std[:, None] - X_std) ** 2, axis=-1))

    # Get maximum possible distance for standardized data
    max_dist = np.sqrt(2 * X_std.shape[1])  # sqrt(2m) from (1 - (-1))^2 terms

    # Normalize by theoretical maximum
    return np.mean(dists) / max_dist


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_gram_matrix(design_matrix):
    """
    Computes the Gram matrix of a given design matrix and plots its heatmap.

    Parameters:
    design_matrix (np.ndarray): An n x m matrix where n is the number of samples and
                                m is the number of features.

    The Gram matrix is computed as:
        G = design_matrix * design_matrix^T
    """
    # Validate input
    if not isinstance(design_matrix, np.ndarray):
        raise ValueError("The design_matrix must be a numpy array.")

    # Compute Gram matrix (n x n)
    gram_matrix = np.dot(design_matrix, design_matrix.T)

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        gram_matrix, annot=True, cmap="viridis", xticklabels=True, yticklabels=True
    )
    plt.title("Gram Matrix Heatmap")
    plt.xlabel("Samples")
    plt.ylabel("Samples")
    plt.savefig("gram_matrix_heatmap.png")


# Example usage:
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_algorithms import NumPyMinimumEigensolver
    from TreeVQA.treevqa_helper import molecule_to_op, solve_groundstate_cupy

    LI_BASE = np.float64(0.472432499)
    H_BASE = np.float64(-1.4172974999999999)
    EQ_BOND_LENGTH = 1.5949
    solver = NumPyMinimumEigensolver()
    # ----------------------------
    # Step 1: Load/Simulate Data (Replace with your Hamiltonians!)
    # ----------------------------
    bond_lengths = np.linspace(0.6, 2.4, 40)  # Replace with your bond lengths
    equilibrium_idx = np.argmin(
        np.abs(bond_lengths - EQ_BOND_LENGTH)
    )  # Update equilibrium index
    atom_strings = [
        f"Li {bond_len * LI_BASE} 0.0 0.0; H {bond_len * H_BASE} 0.0 0.0"
        for bond_len in bond_lengths
    ]
    hamiltonians = [molecule_to_op(atom_str) for atom_str in atom_strings]
    coeffs = [ham[0] for ham in hamiltonians]
    spop = [SparsePauliOp(data=ham[1], coeffs=ham[0]) for ham in hamiltonians]
    sp_matrix = [op.to_matrix() for op in spop]
    ground_states = [solve_groundstate_cupy(op).eigenstate for op in spop]

    # ----------------------------
    # Step 2: Compute Similarity Matrix
    # ----------------------------

    def compute_similarity_fidelity(ground_states):
        n = len(ground_states)
        fidelity = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                state_i = ground_states[i]
                state_j = ground_states[j]
                # Compute the fidelity as the absolute square of the inner product
                # fidelity[i, j] = |<state_i|state_j>|^2
                inner_product = np.abs(np.vdot(state_i, state_j))
                fidelity[i, j] = inner_product
        original_fidelity = fidelity.copy()
        max_fidelity = np.max(fidelity)
        min_fidelity = np.min(fidelity)
        fidelity = (fidelity - min_fidelity) / (max_fidelity - min_fidelity)
        return original_fidelity, fidelity

    def compute_similarity_matrix_original_H(hamiltonians):
        n = len(hamiltonians)
        similarity = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                H_i = hamiltonians[i]
                H_j = hamiltonians[j]
                diff_norm = np.linalg.norm(H_i - H_j, "fro")  # Frobenius norm
                max_norm = max(np.linalg.norm(H_i, "fro"), np.linalg.norm(H_j, "fro"))
                similarity[i, j] = 1 - diff_norm / max_norm
        max_sim = np.max(similarity)
        min_sim = np.min(similarity)
        original_similarity = similarity.copy()
        similarity = (similarity - min_sim) / (max_sim - min_sim)
        return original_similarity, similarity

    def compute_similarity_matrix(hamiltonians):
        n = len(hamiltonians)
        similarity = np.zeros((n, n))
        diff_norm = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                # Flattened coefficient vectors
                vec_i = hamiltonians[i]
                vec_j = hamiltonians[j]

                # L1 norm for difference between vectors
                diff_norm[i, j] = np.linalg.norm(vec_i - vec_j, ord=1)
        # normalize the similarity matrix
        # sigma is the median of the diff_norm
        sigma = np.median(diff_norm) * 2
        similarity = np.exp(-(diff_norm**2) / (2 * sigma**2))
        max_sim = np.max(similarity)
        min_sim = np.min(similarity)
        original_similarity = similarity.copy()
        similarity = (similarity - min_sim) / (max_sim - min_sim)
        return original_similarity, similarity

    original_fidelity_matrix, fidelity_matrix = compute_similarity_fidelity(
        ground_states
    )
    print("done 1")
    original_sim, similarity = compute_similarity_matrix(coeffs)
    print("done 2")
    original_sim_mat, similarity_matrix = compute_similarity_matrix_original_H(
        sp_matrix
    )
    print("done 3")
    below_eq = bond_lengths < bond_lengths[equilibrium_idx]
    above_eq = bond_lengths > bond_lengths[equilibrium_idx]

    # ----------------------------
    # Step 3: Generate and Save Plots
    # ----------------------------
    cmap = cm.plasma
    tick_step = 3  # Show every 2nd bondlength (adjust as needed)

    # --- Plot 1: Bondlengths < Equilibrium ---
    # plt.figure(figsize=(6, 5))
    # sim_below = similarity[below_eq][:, below_eq]
    # bl_below = bond_lengths[below_eq]
    # n_below = len(bl_below)
    # Select every `tick_step` indices and labels
    # indices_below = np.arange(0, n_below, tick_step)
    # labels_below = [f"{bl:.1f}" for bl in bl_below[indices_below]]

    # plt.imshow(sim_below, cmap=cmap, vmin=0, vmax=1)
    # plt.title("Bondlengths < Equilibrium")
    # plt.xlabel("Bondlength (Å)")
    # plt.ylabel("Bondlength (Å)")
    # plt.xticks(indices_below, labels=labels_below, rotation=90)
    # plt.yticks(indices_below, labels=labels_below)
    # plt.colorbar(label="Similarity (1 = identical)")
    # plt.tight_layout()
    # plt.savefig("similarity_below.png", dpi=300)
    # plt.close()

    # # --- Plot 2: Bondlengths > Equilibrium ---
    # plt.figure(figsize=(6, 5))
    # sim_above = similarity[above_eq][:, above_eq]
    # bl_above = bond_lengths[above_eq]
    # n_above = len(bl_above)
    # # Select every `tick_step` indices and labels
    # indices_above = np.arange(0, n_above, tick_step)
    # labels_above = [f"{bl:.1f}" for bl in bl_above[indices_above]]

    # plt.imshow(sim_above, cmap=cmap, vmin=0, vmax=1)
    # plt.title("Bondlengths > Equilibrium")
    # plt.xlabel("Bondlength (Å)")
    # plt.ylabel("Bondlength (Å)")
    # plt.xticks(indices_above, labels=labels_above, rotation=90)
    # plt.yticks(indices_above, labels=labels_above)
    # plt.colorbar(label="Similarity (1 = identical)")
    # plt.tight_layout()
    # plt.savefig("similarity_above.png", dpi=300)
    # plt.close()

    # --- Plot 3: Global Heatmap ---
    # Plot for fidelity_matrix
    plt.figure(figsize=(10, 8))
    bl_global = bond_lengths
    n_global = len(bl_global)
    # Select every `tick_step` indices and labels
    indices_global = np.arange(0, n_global, tick_step)
    labels_global = [f"{bl:.1f}" for bl in bl_global[indices_global]]

    plt.rcParams.update({"font.size": 18})
    plt.rcParams.update({"axes.labelsize": 20})
    plt.rcParams.update({"axes.titlesize": 22})
    plt.rcParams.update({"xtick.labelsize": 18})
    plt.rcParams.update({"ytick.labelsize": 18})
    plt.rcParams.update({"legend.fontsize": 18})

    plt.imshow(fidelity_matrix, cmap=cmap, vmin=0, vmax=1)
    plt.title("Similarity of Ground States (Normalized)", fontsize=22)
    plt.xlabel("Bondlength (\AA)", fontsize=20)
    plt.ylabel("Bondlength (\AA)", fontsize=20)
    plt.xticks(indices_global, labels=labels_global, rotation=90, fontsize=18)
    plt.yticks(indices_global, labels=labels_global, fontsize=18)
    # Mark equilibrium point
    plt.axvline(float(equilibrium_idx), color="r", linestyle="--", linewidth=2)
    plt.axhline(float(equilibrium_idx), color="r", linestyle="--", linewidth=2)
    cbar = plt.colorbar(label="Similarity (1 = identical)")
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label("Similarity (1 = identical)", fontsize=18)
    plt.tight_layout()
    plt.savefig("sim-figs/similarity_fidelity_nor.pdf", dpi=300)
    plt.close()

    # Plot for similarity_matrix (Hamiltonian matrices)
    plt.figure(figsize=(10, 8))
    plt.imshow(similarity_matrix, cmap=cmap, vmin=0, vmax=1)
    plt.title("Similarity of Hamiltonians Matrices (Normalized)", fontsize=22)
    plt.xlabel("Bondlength (\AA)", fontsize=20)
    plt.ylabel("Bondlength (\AA)", fontsize=20)
    plt.xticks(indices_global, labels=labels_global, rotation=90, fontsize=18)
    plt.yticks(indices_global, labels=labels_global, fontsize=18)
    # Mark equilibrium point
    plt.axvline(float(equilibrium_idx), color="r", linestyle="--", linewidth=2)
    plt.axhline(float(equilibrium_idx), color="r", linestyle="--", linewidth=2)
    cbar = plt.colorbar(label="Similarity (1 = identical)")
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label("Similarity (1 = identical)", fontsize=18)
    plt.tight_layout()
    plt.savefig("sim-figs/similarity_mat_nor.pdf", dpi=300)
    plt.close()

    # Plot for similarity (coefficient vectors)
    plt.figure(figsize=(10, 8))
    plt.imshow(similarity, cmap=cmap, vmin=0, vmax=1)
    plt.title(
        "Similarity of Hamiltonian Coefficients in TreeVQA norm space (Normalized)",
        fontsize=22,
    )
    plt.xlabel("Bondlength (\AA)", fontsize=20)
    plt.ylabel("Bondlength (\AA)", fontsize=20)
    plt.xticks(indices_global, labels=labels_global, rotation=90, fontsize=18)
    plt.yticks(indices_global, labels=labels_global, fontsize=18)
    # Mark equilibrium point
    plt.axvline(float(equilibrium_idx), color="r", linestyle="--", linewidth=2)
    plt.axhline(float(equilibrium_idx), color="r", linestyle="--", linewidth=2)
    cbar = plt.colorbar(label="Similarity (1 = identical)")
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label("Similarity (1 = identical)", fontsize=18)
    plt.tight_layout()
    plt.savefig("sim-figs/similarity_vec_nor.pdf", dpi=300)
    plt.close()
    plt.close()
