#!/usr/bin/env bash
set -Eeuo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/lib.sh"
stage_header "Installing PyTorch 2.7.1 and its CUDA 12.8 dependencies"
uv_install --index-url https://download.pytorch.org/whl/cu128 -r "$requirements_dir/torch.txt"
