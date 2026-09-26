#!/usr/bin/env bash

readonly venv=/opt/workspace-python
readonly stage_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly requirements_dir="${stage_dir%/install-python-stack}/requirements"

stage_header() {
  printf '==> %s\n' "$*"
}

uv_install() {
  uv pip install --python "$venv/bin/python" --only-binary :all: \
    --constraint "$requirements_dir/constraints.txt" "$@"
}
