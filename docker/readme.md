Linux/amd64 Docker image for the paper [Residual Learning in Empirical Asset Pricing](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7528558).
Mount the repository and your licensed data.

| | Version |
|---|---|
| Ubuntu | 24.04 |
| Python / uv | 3.13.5 / 0.7.19 |
| PyTorch | 2.7.1+cu128 |
| PyTorch cuBLAS / cuDNN | 12.8.3.14 / 9.7.1.26 |
| NumPy / pandas | 2.1.2 / 2.3.1 |
| SciPy / matplotlib | 1.16.0 / 3.10.3 |
| statsmodels / ml-collections | 0.14.5 / 1.1.0 |
| JupyterLab / ipykernel | 4.4.4 / 6.29.5 |
| TensorBoard / openpyxl | 2.19.0 / 3.1.5 |

| | Sample Command |
|---|---|
| Build | `docker build --platform linux/amd64 -t resassetpricing:latest docker` |
| Run, With NVIDIA GPU | `docker run --rm -it --gpus all -p 127.0.0.1:8888:8888 -v "${PWD}:/workspace" resassetpricing:latest` |
| Run, No NVIDIA GPU | `docker run --rm -it --platform linux/amd64 -e NVIDIA_VISIBLE_DEVICES=void -p 127.0.0.1:8888:8888 -v "${PWD}:/workspace" resassetpricing:latest` |
