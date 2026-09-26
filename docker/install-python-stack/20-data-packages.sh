#!/usr/bin/env bash
set -Eeuo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/lib.sh"
stage_header "Installing the project's numerical and data packages"
uv_install --index-url https://pypi.org/simple -r "$requirements_dir/data.txt"
