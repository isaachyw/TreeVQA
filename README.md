# TreeVQA

> **A Tree-Structured Execution Framework for Shot Reduction in Variational Quantum Algorithms**

This repository contains the official implementation and artifact for reproducing the experiments in the paper.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17945742.svg)](https://doi.org/10.5281/zenodo.17945742)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2512.12068-b31b1b.svg)](https://arxiv.org/abs/2512.12068)


---

## 📋 Requirements

- **Python**: >= 3.12
- **Julia**: Required for `juliacall` dependency
- **LaTeX**: Required for generating publication-quality plots (`texlive` or equivalent)
- **Package Manager**: [uv](https://astral.sh/uv) (automatically installed if missing)


## 🚀 Quick Start

Clone the repository and run the automated scripts in sequence:

```bash
git clone https://github.com/isaachyw/TreeVQA.git
cd TreeVQA
chmod +x scripts/*
```

### Automated Workflow

| Step | Script | Description | Duration |
|------|--------|-------------|----------|
| 0 | `./scripts/0_setup.sh` | Install dependencies & setup environment | ~5 min |
| 1 | `./scripts/1_ground_truth.sh` | Compute ground truth energies | ~1 hour |
| 2 | `./scripts/2_minimal_example.sh` | Run minimal TreeVQA example | ~10 min |
| 3 | `./scripts/3_analysis_plot.sh` | Generate analysis plots | ~5 min |

```bash
./scripts/0_setup.sh
./scripts/1_ground_truth.sh
./scripts/2_minimal_example.sh
./scripts/3_analysis_plot.sh
```

---

## 📁 Project Structure

```
treevqa/
├── TreeVQA/                # Core TreeVQA implementation
│   ├── application/        # VQE applications (molecules, Ising, MaxCut, etc.)
│   ├── optimizer/          # Custom optimizers (COBYLA, SPSA with TreeVQA)
│   ├── vqa/                # VQE cluster execution utilities
│   └── clapton/            # Clifford-based circuit optimization
├── config/                 # Experiment configuration files
│   ├── cobyla/             # COBYLA optimizer configs - fig 9
│   ├── spsa/               # SPSA optimizer configs - fig 5
│   └── noisy/              # Noisy simulation configs - under construction
├── ground-state/           # Ground state energy computation
├── plot_util/              # Plotting and analysis scripts
├── scripts/                # Automated workflow scripts
├── main.py                 # Main entry point
├── batch_ne.py             # Batch experiment runner
└── single_vqe.py           # baseline VQE experiment runner
```

---

## Main Experiments

The main experiments for **Figure 5** and **Figure 9** require cluster resources or long-running jobs. These are not fully automated.

### [SLURM](https://slurm.schedmd.com/documentation.html) Cluster Submission

1. Copy and configure the SLURM settings:
   ```bash
   cp slurm_config.json.example slurm_config.json
   # Edit slurm_config.json with your cluster settings
   ```

2. Generate and submit batch jobs:
   ```bash
   python3 batch_ne.py config/your-config.json --slurm
   ```
or just generate the commands to a concurrent*.sh file so that you can run the configs on your own.


For running without SLURM, the script generates a `concurrent_*.sh` file that you can execute directly:

```bash
python3 batch_ne.py config/your-config.json
./concurrent_*.sh
```

---

## ⚠️ Known Issues

### JuliaCall Deadlock on SLURM

When running on SLURM clusters, `juliacall` may deadlock during Julia package resolution. 

**Workaround**: Comment out lines 321–329 in `.venv/lib64/python3.12/site-packages/juliacall/deps.py`:

```python
# try:
#     lock.acquire(timeout=3)
# except TimeoutError:
#     log(
#         f"Waiting for lock on {lock_file} to be freed. This normally means that"
#         " another process is resolving. If you know that no other process is"
#         " resolving, delete this file to proceed."
#     )
#     lock.acquire()
```

---

## 📖 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{treevqa2025,
  title={TreeVQA: A Tree-Structured Execution Framework for Shot Reduction in Variational Quantum Algorithms},
  year={2025}
}
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
