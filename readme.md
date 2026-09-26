# Residual Learning in Empirical Asset Pricing

Source code for the paper [Residual Learning in Empirical Asset Pricing](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7528558).
Check [readme.ipynb](readme.ipynb) to get a quick start.
Run `./run.sh` (macOS/Linux/WSL) or `.\run.ps1` (Windows PowerShell) to set up
the environment and interactively select experiments.

## Quick Start

### Run Sample Experiment

Execute `./readme.ipynb`

### Run All Experiments

```shell
./run.sh
```

## Compatibility

|  | Device | Supported | Recommended |
|---|---|---|---|
| Linux | NVIDIA GPU | Y | Y |
| Windows / WSL2 | NVIDIA GPU | Y | |
| Linux / Windows x64 | CPU | Y | |
| Native Windows x64 | NVIDIA GPU | Y | |
| macOS | Apple Silicon | Y | |

| | Storage | Required |
|---|---|---|
| Source SAS Data | ~41 GiB | |
| Clean Inputs | ~4.4 GiB | Y |
| Annual Splits (Optional) | ~56.6 GiB | |
| One Specification Output | 1–3 GiB | Y |
| Docker Image | ~11 GiB | |
| Recommended Total Budget | ~350 GiB | |

## Environment

Use `uv` or Docker. Run commands from the repository root.

### With Docker

See [Software Versions](docker/readme.md).

| | Sample Command |
|---|---|
| Build | `docker build --platform linux/amd64 -t resassetpricing:latest docker` |
| Run, With NVIDIA GPU | `docker run --rm -it --gpus all -p 127.0.0.1:8888:8888 -v "${PWD}:/workspace" resassetpricing:latest` |
| Run, No NVIDIA GPU | `docker run --rm -it --platform linux/amd64 -e NVIDIA_VISIBLE_DEVICES=void -p 127.0.0.1:8888:8888 -v "${PWD}:/workspace" resassetpricing:latest` |

### With `uv`

| Step | Command |
|---|---|
| Python | `uv venv --python 3.13.5` |
| Dependencies | `uv pip install -r docker/requirements/common.txt -r docker/requirements/data.txt` |
| NVIDIA — Linux / Windows | `uv pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128` |
| CPU — Linux / Windows | `uv pip install -r docker/requirements/torch-cpu.txt --index-url https://download.pytorch.org/whl/cpu` |
| macOS | `uv pip install -r docker/requirements/torch-macos.txt` |
| Native Windows TUI | `uv pip install -r docker/requirements/windows.txt` |

## Cost

Measured on NVIDIA H100 PCIe 80 GB; ten seeds.

| | | Time |
|---|---|---|
| Quick Start | Train | ~84 min measured |
| Quick Start | Predict | ~24 s measured |
| One Seed, Latest Year | Inference | ~100 ms estimated |
| Full Replication | | ~300 GPU-h estimated |

Download the pre-trained models from [OneDrive](https://hkustgz-my.sharepoint.com/:u:/g/personal/dpeng965_connect_hkust-gz_edu_cn/IQDQXtmx6QfSTKf0-bcg3NnIAS5kUfk4rPii6_Me0nbXmwo?e=EvgJez) to skip the training process.
The licensed input data are still required; the checkpoint archive contains pre-trained models only.

## Source Data

Obtain the licensed data from WRDS with [JKP](https://github.com/bkelly-lab/ReplicationCrisis) scripts.

Other source data are publicly accessible and included in `./source_data`.

| | Contents | Public | Link |
|---|---|---|---|
| `source_data/datashare_with_return.pkl` | U.S. Panel | N | [JKP](https://github.com/bkelly-lab/ReplicationCrisis) |
| `source_data/PredictorData2024.xlsx` | Goyal macro, 2024 | Y | [Amit Goyal](https://sites.google.com/view/agoyal145) |
| `source_data/F-F_Research_Data_5_Factors_2x3.csv` | French factors | Y | [Kenneth French](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) |

## Use Your Own Data

The predictor DataFrame `clean_X.pkl` should have the following structure:

| | | market_equity | be_me | ... | dfy | svar |
|---:|---|---:|---:|:---:|---:|---:|
| **permno** | **DATE** | | | | | |
| 10001 | 2023-01-31 | -0.60 | 0.20 | ... | 1.10 | 0.002 |
| 10001 | 2023-02-28 | -0.40 | 0.30 | ... | 1.05 | 0.003 |
| ... | ... | ... | ... | ... | ... | ... |
| 10099 | 2023-01-31 | 0.40 | -0.20 | ... | 1.10 | 0.002 |
| 10099 | 2023-02-28 | 0.60 | -0.10 | ... | 1.05 | 0.003 |

The response Series `clean_y.pkl` should have the following structure:

| | | RET |
|---:|---|---:|
| **permno** | **DATE** | |
| 10001 | 2023-01-31 | 0.012 |
| 10001 | 2023-02-28 | -0.008 |
| ... | ... | ... |
| 10099 | 2023-01-31 | 0.015 |
| 10099 | 2023-02-28 | -0.003 |

Note, X contains predictors available at t, and y is the realized excess return in month t+1.
