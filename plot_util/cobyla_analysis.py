import numpy as np
from efficiency_compare import Result_analysis
import os
import shutil

PAULI_MAPS = {
    "LiH": 496,
    "BeH2": 810,
    "HF": 631,
    "H2O": 1086,
    "HF-uccsd": 631,
    "H2-uccsd": 15,
    "MaxCut-10": 45,
    "XXZ-9": 96,  # 18 qubits
    "SL-3": 24,  # 8 qubits
    "LL-6": 32,
    "LL-5": 26,
    "XXZ-6": 60,
    "XXZ-Model": 32,
    "Transverse-Field-Model": 26,
    "H_2-UCCSD": 15,
}


def main():
    os.makedirs("paper-fig/fig9", exist_ok=True)
    # Example directories
    result_dirs = [
        "../results-cobyla/LiH/1.4:1.67:0.03/VQE15000_W400_WA800_T0.3_-2e-05_HEA",
        "../results-cobyla/HF/0.83:1.1:0.03/VQE8000_W300_WA800_T0.3_-0.005_HEA",
        "../results-cobyla/BeH2/1.2:1.47:0.03/VQE15000_W600_WA1000_T0.3_-1e-05_HEA",
        "../results-cobyla/IsingModel5node/0.96:1.05:0.01/VQE40000_W200_WA400_T(0.3, -5e-05)_HEA",
        "../results-cobyla/H2/0.74:0.83:0.02/VQE400_W40_WA80_T0.3_-0.0001_UCCSD",
        "../results-cobyla/xxz5node/0.96:1.05:0.01/VQE60000_W300_WA800_T(0.3, -9e-05)_HEA",
    ]
    molecules = [
        "LiH",
        "HF",
        "BeH2",
        "Transverse-Field-Model",
        "H2-uccsd",
        "XXZ-Model",
    ]
    # store the data.json
    separate_vqe_dirs = [
        "../results-cobyla/LiH",
        "../results-cobyla/HF",
        "../results-cobyla/BeH2",
        "../results-cobyla/IsingModel5node",
        "../results-cobyla/H2",
        "../results-cobyla/xxz5node",
    ]
    save_dirs = result_dirs
    paulis = [PAULI_MAPS[molecule] for molecule in molecules]
    budgets = [
        np.arange(3000, 15000, 50),
        np.arange(4000, 10000, 50),
        np.arange(7000, 20000, 50),
        np.arange(4000, 40000, 50),
        np.arange(100, 800, 10),
        np.arange(1000, 60000, 50),
    ]
    fidelitys = [
        np.arange(0.8, 0.96, 0.0001),
        np.arange(0.8, 0.98, 0.0001),
        np.arange(0.6, 0.952, 0.0001),
        np.arange(0.6, 0.76, 0.0001),
        np.arange(0.6, 1, 0.0001),
        np.arange(0.6, 0.795, 0.0001),
    ]

    # Pre-zip all arrays for selective plotting
    all_data = list(
        zip(
            result_dirs,
            molecules,
            separate_vqe_dirs,
            save_dirs,
            paulis,
            budgets,
            fidelitys,
        )
    )

    # Now you can select which items to plot
    for (
        result_dir,
        molecule,
        separate_vqe_dir,
        save_dir,
        pauli_num,
        budget,
        fidelity,
    ) in all_data:
        num_batch = 4096
        optimizer_constant = 2
        # Instantiate the analysis object
        analysis = Result_analysis(
            result_dir=result_dir,
            seperate_vqe_dir=separate_vqe_dir,
            save_dir=save_dir,
            num_pauli_str=pauli_num,
            num_batch=num_batch,
            optimizer_constant=optimizer_constant,
            molecule_name=molecule,
        )

        analysis.visualize_shots_for_various_fidelity_avg(fidelity)
        # copy the figure to the paper-fig/fig5 directory
        shutil.copy(
            analysis.save_dir + f"/{molecule}shots_vs_avg_fidelity_threshold.pdf",
            "paper-fig/fig9/cobyla_" + molecule + "shots_vs_fidelity_threshold.pdf",
        )


if __name__ == "__main__":
    main()
