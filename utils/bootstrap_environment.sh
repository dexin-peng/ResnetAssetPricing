# Sourced by run.sh after argument validation. Compatible with macOS Bash 3.2.

bootstrap_detect_platform() {
  BOOTSTRAP_SYSTEM="$(uname -s)"
  BOOTSTRAP_ARCH="$(uname -m)"
  BOOTSTRAP_PROFILE=cpu
  case "${BOOTSTRAP_SYSTEM}" in
    Darwin)
      BOOTSTRAP_PROFILE=macos
      printf 'System: macOS | Hardware: %s | Backend: Apple MPS / CPU\n' "${BOOTSTRAP_ARCH}"
      ;;
    Linux)
      local label=Linux gpu_info
      case "$(uname -r)" in *[Mm]icrosoft*) label='Linux / WSL' ;; esac
      if command -v nvidia-smi >/dev/null 2>&1 && gpu_info="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)" && [[ -n "${gpu_info}" ]]; then
        BOOTSTRAP_PROFILE=cuda
        printf 'System: %s | Hardware: %s, %s | Backend: NVIDIA CUDA\n' "${label}" "${BOOTSTRAP_ARCH}" "${gpu_info}"
      else
        printf 'System: %s | Hardware: %s | Backend: CPU (no usable NVIDIA driver detected)\n' "${label}" "${BOOTSTRAP_ARCH}"
      fi
      ;;
    MINGW*|MSYS*|CYGWIN*) die 'Use .\run.ps1 from PowerShell on native Windows, or ./run.sh inside WSL.' ;;
    *) die "Unsupported system: ${BOOTSTRAP_SYSTEM}. See readme.md." ;;
  esac
  if [[ "${DEVICE_SPEC}" == cpu && "${BOOTSTRAP_SYSTEM}" != Darwin ]]; then
    BOOTSTRAP_PROFILE=cpu
    printf 'CPU explicitly selected.\n'
  elif [[ -z "${DEVICE_SPEC}" && "${BOOTSTRAP_PROFILE}" == cuda && -f "${PROJECT_ROOT}/.venv/.launcher-backend" ]]; then
    if [[ "$(cat "${PROJECT_ROOT}/.venv/.launcher-backend")" == cpu ]]; then
      BOOTSTRAP_PROFILE=cpu
      printf 'Reusing the CPU environment selected during setup.\n'
    fi
  fi
}

bootstrap_install_dependencies() {
  "${BOOTSTRAP_UV}" pip install --python "${PYTHON_BIN}" "$@" \
    -r "${PROJECT_ROOT}/docker/requirements/common.txt" \
    -r "${PROJECT_ROOT}/docker/requirements/data.txt" || return $?
  local requirement=torch-macos.txt
  local index_url=https://pypi.org/simple
  case "${BOOTSTRAP_PROFILE}" in
    cuda) requirement=torch.txt; index_url=https://download.pytorch.org/whl/cu128 ;;
    cpu) requirement=torch-cpu.txt; index_url=https://download.pytorch.org/whl/cpu ;;
  esac
  "${BOOTSTRAP_UV}" pip install --python "${PYTHON_BIN}" "$@" \
    --index-url "${index_url}" -r "${PROJECT_ROOT}/docker/requirements/${requirement}"
}

bootstrap_confirm() {
  local reply
  while true; do
    printf '%s [Y/n] ' "$1"
    IFS= read -r reply || return 1
    case "${reply}" in
      ''|y|Y|yes|YES) return 0 ;;
      n|N|no|NO) return 1 ;;
      *) printf 'Please enter y or n.\n' ;;
    esac
  done
}

bootstrap_retry() {
  local label="$1"
  shift
  until "$@"; do
    printf '\n%s failed. Check the output above, network/proxy settings and free disk space.\n' "${label}"
    bootstrap_confirm 'Retry this step?' || die "Setup stopped. Run ./run.sh again to resume."
  done
}

bootstrap_find_uv() {
  local candidate
  BOOTSTRAP_UV="$(command -v uv || true)"
  if [[ -n "${BOOTSTRAP_UV}" ]] && "${BOOTSTRAP_UV}" --version >/dev/null 2>&1; then
    return 0
  fi
  for candidate in "${HOME}/.local/bin/uv" "${HOME}/.cargo/bin/uv" /opt/homebrew/bin/uv /usr/local/bin/uv; do
    if [[ -x "${candidate}" ]] && "${candidate}" --version >/dev/null 2>&1; then
      BOOTSTRAP_UV="${candidate}"
      return 0
    fi
  done
  return 1
}

bootstrap_install_uv() {
  local installer status=0
  installer="$(mktemp "${TMPDIR:-/tmp}/resassetpricing-uv.XXXXXX")" || return 1
  if command -v curl >/dev/null 2>&1; then
    curl --proto '=https' --tlsv1.2 --fail --location --show-error \
      --connect-timeout 20 --max-time 180 https://astral.sh/uv/install.sh -o "${installer}" || status=$?
  elif command -v wget >/dev/null 2>&1; then
    wget --timeout=30 -O "${installer}" https://astral.sh/uv/install.sh || status=$?
  else
    printf 'Install curl or wget using your system package manager, then retry.\n'
    status=1
  fi
  if (( status == 0 )); then
    env UV_INSTALL_DIR="${HOME}/.local/bin" UV_NO_MODIFY_PATH=1 sh "${installer}" || status=$?
  fi
  rm -f -- "${installer}"
  return "${status}"
}

bootstrap_probe() {
  command -v "${PYTHON_BIN}" >/dev/null 2>&1 || {
    printf 'No Python interpreter found: %s\n' "${PYTHON_BIN}"
    return 10
  }
  "${PYTHON_BIN}" "${PROJECT_ROOT}/utils/check_environment.py" --project-root "${PROJECT_ROOT}" --profile "${BOOTSTRAP_PROFILE}" "$@"
}

bootstrap_environment() {
  local status=0 project_python="${PROJECT_ROOT}/.venv/bin/python" backup force_cpu=0
  local -a install_args=(--no-python-downloads)
  printf '\nChecking the Python environment...\n'
  bootstrap_probe || status=$?
  if (( status == 0 )); then
    return 0
  fi

  if [[ ! -t 0 || ! -t 1 ]]; then
    die "Environment setup needs an interactive terminal. Run ./run.sh --setup in Terminal, then rerun this command. No packages were installed."
  fi
  if (( DRY_RUN )); then
    die "The environment is not ready. Run ./run.sh --setup first; --dry-run does not install anything."
  fi
  if [[ "${BOOTSTRAP_SYSTEM}" == Darwin && "${BOOTSTRAP_ARCH}" != arm64 ]]; then
    if [[ "$(sysctl -in sysctl.proc_translated 2>/dev/null || true)" == 1 ]]; then
      die "This terminal runs under Rosetta. Reopen Terminal without 'Open using Rosetta', then run ./run.sh."
    fi
    die "The pinned native PyTorch package requires Apple Silicon. On an Intel Mac use the Linux/amd64 CPU Docker setup in readme.md."
  fi
  if [[ "${BOOTSTRAP_SYSTEM}" == Linux && "${BOOTSTRAP_ARCH}" != x86_64 && "${BOOTSTRAP_ARCH}" != aarch64 ]]; then
    die "Automatic setup supports Linux x86_64/aarch64. Prepare a compatible environment and use --no-setup on ${BOOTSTRAP_ARCH}."
  fi

  if [[ "${BOOTSTRAP_PROFILE}" == cuda ]]; then
    printf '\nNVIDIA hardware detected. CUDA 12.8 PyTorch needs a compatible NVIDIA driver.\n'
    if ! bootstrap_confirm 'Install the CUDA build? (n selects the smaller CPU build)'; then
      BOOTSTRAP_PROFILE=cpu
      force_cpu=1
      install_args+=(--reinstall-package torch)
    fi
  fi

  printf '\nSetup will use %s\n' "${PROJECT_ROOT}/.venv"
  printf 'It can download uv, Python 3.13.5, PyTorch and the notebook/data dependencies.\n'
  printf 'An internet connection and several GB of free space are needed.\n'
  printf 'System Python and other active environments will be left unchanged.\n'
  printf 'For a custom environment, use ./run.sh --no-setup with your usual arguments.\n'
  bootstrap_confirm 'Set up or repair the project environment?' || die "Setup cancelled."

  if ! bootstrap_find_uv; then
    printf '\nuv is missing. Install it to %s/.local/bin using https://astral.sh/uv/install.sh.\n' "${HOME}"
    printf 'The installer will not edit your shell profile.\n'
    bootstrap_confirm 'Install uv?' || die "Setup cancelled."
    bootstrap_retry 'uv installation' bootstrap_install_uv
    bootstrap_find_uv || die "uv installation finished but uv could not be run. Check ${HOME}/.local/bin/uv."
  fi
  printf 'Using %s\n' "${BOOTSTRAP_UV}"

  # Repair only the project environment, even when an external Python was selected.
  PYTHON_BIN="${project_python}"
  status=0
  bootstrap_probe --python-only --project-venv || status=$?
  if (( status != 0 )) || [[ -L "${PROJECT_ROOT}/.venv" ]]; then
    if [[ -e "${PROJECT_ROOT}/.venv" || -L "${PROJECT_ROOT}/.venv" ]]; then
      backup="${PROJECT_ROOT}/.venv.backup.$(date +%Y%m%d-%H%M%S).$$"
      printf 'The existing .venv needs replacing. It will be preserved at:\n%s\n' "${backup}"
      bootstrap_confirm 'Back up the existing environment and create a new one?' || die "Setup cancelled."
      mv -- "${PROJECT_ROOT}/.venv" "${backup}" || die "Cannot back up .venv. Check directory permissions."
    fi
    bootstrap_retry 'Python environment creation' "${BOOTSTRAP_UV}" venv --python 3.13.5 --managed-python "${PROJECT_ROOT}/.venv"
  fi

  status=0
  bootstrap_probe || status=$?
  if (( force_cpu )); then status=20; fi
  if (( status != 0 )); then
    # Reinstall if the metadata is present but imports are broken; ordinary
    # missing/version-mismatched packages only need the normal resolver.
    if (( status != 20 )); then
      install_args+=(--reinstall)
    fi
    while true; do
      bootstrap_retry 'Dependency installation' bootstrap_install_dependencies "${install_args[@]}"
      if bootstrap_probe; then
        break
      fi
      bootstrap_confirm 'Environment verification failed. Reinstall the dependencies and retry?' || die "Setup stopped. See the diagnostic above."
      install_args=(--no-python-downloads --reinstall)
    done
  fi
  printf '%s\n' "${BOOTSTRAP_PROFILE}" > "${PROJECT_ROOT}/.venv/.launcher-backend"
  printf 'Project environment ready: %s\n\n' "${PYTHON_BIN}"
}
