# Guide for Agents

This file provides guidance for external AI agents working with *Residual Learning in Empirical Asset Pricing*.

## Entry Points and System Compatibility

### Scope of Work

- Use `uv` or the corresponding virtual environment's interpreter for Python commands. Keep validation proportional to the task, distinguishing static checks, small-sample validation, and full experimental replication.

### Directories and Entry Points

| Path | Purpose |
|---|---|
| `readme.md` | Installation, resource budgets, data sources |
| `readme.ipynb` | Data inspection, preprocessing, training, prediction, economic evaluation |
| `run.sh` | Opens the TUI without arguments; also supports command-line options and saved plans |
| `main.py` | Training, prediction, and economic evaluation for one configuration and seed |
| `config/` | Widths, depths, sample windows, and output paths for the four model families |
| `model/` | NN, ResNet, NN+, and ResNet+ architectures |
| `run/` | Data loading, training, prediction, economic metrics |
| `source_data/` | Small public files and licensed data obtained by the user |
| `tmp/` | Cleaned caches, annual data splits, temporary files; excluded from Git |
| `asset/<asset_prefix>/` | Checkpoints, predictions, economic results, and logs; excluded from Git |
| `utils/` | Data preparation, device selection, TUI, progress display, and other utilities |
| `utils/paper_source/` | DataFrame tables, figures, and numerical previews in `exhibits.ipynb` |
| `docker/` | Public image build files and the single location for dependency files |

Present tables as DataFrames and in notebooks. Specify return units, dates, samples, weighting, and aggregation methods. Report any differences between older cached results and current calculations.

### Installation and Startup

Tailor guidance to the computing resources available to the user. Prefer Docker for the full reference environment; use native macOS with `uv` for Apple GPU acceleration. Refer to `docker/readme.md` and `docker/requirements/` for specific versions.

| Environment | Device and requirements |
|---|---|
| Linux + NVIDIA | CUDA; install a compatible driver and CUDA-enabled PyTorch |
| Windows + WSL2 | Follow the Linux setup; GPU use requires WSL2 CUDA support |
| Native Windows | Explain Python CPU/CUDA use and the Bash TUI separately; do not promise that PowerShell can run `.sh` directly; WSL2 is recommended for the TUI |
| Native macOS | Select MPS when available; otherwise notify the user and use CPU |
| No available GPU | Use CPU; do not continue generating commands with a hardcoded CUDA device |
| Linux container | GPU use requires NVIDIA container support and `--gpus all`; the TUI requires `-it` |

- Linux Docker images on macOS cannot provide native MPS acceleration.
- The TUI runs one task at a time on CPU and up to ten concurrently on MPS; CUDA concurrency depends on available GPU memory and the plan settings.
- Shell loops in the notebook are not subject to the TUI's concurrency limits. The current training and prediction examples can launch ten seeds concurrently. On devices with limited GPU or system memory, processes may be terminated because of insufficient memory.

### Pretrained Models

When users have limited computing resources but sufficient storage, recommend the project's publicly available pretrained models. Use the download link published in `readme.md`.

## Random Seed

- With the following version configuration, results can be reproduced exactly at float32 precision. Exact agreement at float32 precision is not guaranteed if any of these versions differs.

| Component | Reference setting |
|---|---|
| Platform | Linux |
| Python | 3.13.5 |
| PyTorch / CUDA wheel | 2.7.1+cu128 / CUDA 12.8 |
| PyTorch cuBLAS | `nvidia-cublas-cu12==12.8.3.14` |
| PyTorch cuDNN | `nvidia-cudnn-cu12==9.7.1.26` |
| NVIDIA architecture | H100, compute capability `(9, 0)` |

## Citation
