#!/usr/bin/env bash
set -Eeuo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/lib.sh"

stage_header "Checking dependencies and registering the notebook kernel"
uv pip check --python "$venv/bin/python"
"$venv/bin/python" -m ipykernel install --sys-prefix --name python3 \
  --display-name "Python 3 (ResAssetPricing)"
