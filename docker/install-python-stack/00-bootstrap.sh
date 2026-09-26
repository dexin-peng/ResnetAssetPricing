#!/usr/bin/env bash
set -Eeuo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/lib.sh"
python_version="${1:-3.13.5}"

stage_header "Preparing Python $python_version"
uv python install "$python_version"
uv venv --python "$python_version" "$venv"
"$venv/bin/python" --version
