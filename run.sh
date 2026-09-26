#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
MAIN_PY="${PROJECT_ROOT}/main.py"
RUN_TUI_PY="${PROJECT_ROOT}/utils/tui_launcher.py"
select_python() {
  # Select only after OS/hardware detection. Respect an explicit interpreter.
  if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
      PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
    elif [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
      PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
      PYTHON_BIN=python3
    else
      PYTHON_BIN=python
    fi
  fi
}
DEVICE_HELPER="${PROJECT_ROOT}/utils/runtime_device.py"
MAX_HIDDEN_LAYER_COUNT=20

EXPERIMENT=""
NUM_GPUS="1"
DEVICE_SPEC=""
LAYERS_SPEC=""
DEFAULT_START_ENSEMBLE=0
DEFAULT_MAX_ENSEMBLE=9
ENSEMBLE_RANGE="${DEFAULT_START_ENSEMBLE}:${DEFAULT_MAX_ENSEMBLE}"
DRY_RUN=0
TUI_MODE=0
PLAN_PATH=""
RUN_ECON=1
GROUP_ABLATION_ONLY=0
SETUP_ONLY=0
SKIP_SETUP=0

START_ENSEMBLE="${DEFAULT_START_ENSEMBLE}"
MAX_ENSEMBLE="${DEFAULT_MAX_ENSEMBLE}"
USE_PRESPLIT=0

GPU_DEVICES=()
ACTIVE_PIDS=()
ACTIVE_LABELS=()
AVAILABLE_SLOT=""

PROGRESS_BAR_PY="${PROJECT_ROOT}/utils/progress_bar.py"
PROGRESS_SESSION_ID=""
PROGRESS_ENABLED=0
PROGRESS_ROOT="${ResAssetPricing_PROGRESS_ROOT:-}"
PROGRESS_FOOTER_LINES="${ResAssetPricing_PROGRESS_FOOTER_LINES:-0}"

DRY_RUN_NEXT_SLOT=0

COMMON_TRAINING_ARGS=(
  --epochs 100
  --batch_size 10000
  --early_stopping_patience 5
  --l2
  --weight_decay 1e-4
  --initial_lr 1e-4
)
PLAIN_RESNET_NN_TRAINING_ARGS=(
  --epochs 100
  --batch_size 10000
  --early_stopping_patience 5
  --l1
  --l1_lambda 1e-4
  --initial_lr 1e-4
)
REPORT_GROUPS=(ALL TOP80% BOTTOM80%)
ECON_WEIGHT_ARGS=(--econ_weight_type vw ew)

usage() {
  cat <<'EOF'
Usage:
  ./run.sh                        # TUI; automatically selects CUDA, MPS or CPU
  ./run.sh --tui [--dry-run]
  ./run.sh --setup                # check/repair environment, then exit
  ./run.sh --no-setup --tui        # use an environment you manage yourself
  ./run.sh --plan /path/to/plan.json [--no-econ] [--dry-run]
  ./run.sh --experiment <name> [--num-gpus N | --devices 0,1,2,3]
           [--layers spec] [--ensemble-range START:END]
           [--devices cpu | --devices mps]
           [--no-econ | --group-ablation-only]
           [--dry-run] [--list-experiments]

Paper-facing experiments:
  resnetps1 .. resnetps4 Seeded ResNet+ families
  nnps1 .. nnps4         Seeded NN+ families
  resnets1 .. resnets4   Seeded ResNet families
  nns1 .. nns4           Seeded NN families

Group ablation (all four model families):
  --group-ablation-only, or TUI r -> 3, uses existing checkpoints without training.
  Also includes the same-family shallow anchor once per selected width (depth = width).
  Anchors use the same ensemble range and all 13 zeroed themes as the selected depths.
  Saves econ results and stock_predictions.pkl.gz for each of the 13 themes.
  Existing econ-only results rerun until their stock predictions are also saved.
EOF
}

experiment_role_text() {
  local experiment="$1"
  parse_experiment_metadata "${experiment}" || return 1

  case "${PARSED_EXPERIMENT_KIND}" in
    nn)
      printf 'NN (s%s)' "${PARSED_EXPERIMENT_SEED}"
      ;;
    nnp)
      printf 'NN+ (s%s)' "${PARSED_EXPERIMENT_SEED}"
      ;;
    resnet)
      printf 'ResNet (s%s)' "${PARSED_EXPERIMENT_SEED}"
      ;;
    resnetp)
      printf 'ResNet+ (s%s)' "${PARSED_EXPERIMENT_SEED}"
      ;;
    *)
      return 1
      ;;
  esac
}

list_experiments() {
  printf '%-20s %-38s %s\n' 'Experiment' 'Role' 'Valid hidden layers'
  local experiment
  for experiment in \
    resnetps1 resnetps2 resnetps3 resnetps4 \
    nnps1 nnps2 nnps3 nnps4 \
    resnets1 resnets2 resnets3 resnets4 \
    nns1 nns2 nns3 nns4
  do
    parse_experiment_metadata "${experiment}" || die "Unsupported experiment in registry: ${experiment}"
    printf '%-20s %-38s %s\n' \
      "${experiment}" \
      "$(experiment_role_text "${experiment}")" \
      "${PARSED_EXPERIMENT_SEED}:${MAX_HIDDEN_LAYER_COUNT}"
  done
}

die() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

assert_file() {
  [[ -e "$1" ]] || die "Required file not found: $1"
}

PARSED_EXPERIMENT_KIND=""
PARSED_EXPERIMENT_SEED=""

parse_experiment_metadata() {
  local experiment="$1"
  PARSED_EXPERIMENT_KIND=""
  PARSED_EXPERIMENT_SEED=""

  if [[ "${experiment}" =~ ^nns([1-4])$ ]]; then
    PARSED_EXPERIMENT_KIND="nn"
    PARSED_EXPERIMENT_SEED="${BASH_REMATCH[1]}"
    return 0
  fi
  if [[ "${experiment}" =~ ^nnps([1-4])$ ]]; then
    PARSED_EXPERIMENT_KIND="nnp"
    PARSED_EXPERIMENT_SEED="${BASH_REMATCH[1]}"
    return 0
  fi
  if [[ "${experiment}" =~ ^resnets([1-4])$ ]]; then
    PARSED_EXPERIMENT_KIND="resnet"
    PARSED_EXPERIMENT_SEED="${BASH_REMATCH[1]}"
    return 0
  fi
  if [[ "${experiment}" =~ ^resnetps([1-4])$ ]]; then
    PARSED_EXPERIMENT_KIND="resnetp"
    PARSED_EXPERIMENT_SEED="${BASH_REMATCH[1]}"
    return 0
  fi

  return 1
}

parse_args() {
  if (( $# == 0 )); then
    TUI_MODE=1
    return 0
  fi

  while (( $# > 0 )); do
    case "$1" in
      --experiment)
        [[ $# -ge 2 ]] || die "Missing value after --experiment"
        EXPERIMENT="$2"
        shift 2
        ;;
      --num-gpus)
        [[ $# -ge 2 ]] || die "Missing value after --num-gpus"
        NUM_GPUS="$2"
        shift 2
        ;;
      --devices)
        [[ $# -ge 2 ]] || die "Missing value after --devices"
        DEVICE_SPEC="$2"
        shift 2
        ;;
      --layers)
        [[ $# -ge 2 ]] || die "Missing value after --layers"
        LAYERS_SPEC="$2"
        shift 2
        ;;
      --ensemble-range)
        [[ $# -ge 2 ]] || die "Missing value after --ensemble-range"
        ENSEMBLE_RANGE="$2"
        shift 2
        ;;
      --dry-run)
        DRY_RUN=1
        shift
        ;;
      --no-econ|--skip-econ)
        RUN_ECON=0
        shift
        ;;
      --group-ablation-only)
        GROUP_ABLATION_ONLY=1
        RUN_ECON=0
        shift
        ;;
      --tui)
        TUI_MODE=1
        shift
        ;;
      --setup)
        SETUP_ONLY=1
        shift
        ;;
      --no-setup)
        SKIP_SETUP=1
        shift
        ;;
      --plan)
        [[ $# -ge 2 ]] || die "Missing value after --plan"
        PLAN_PATH="$2"
        shift 2
        ;;
      --list-experiments)
        list_experiments
        exit 0
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        die "Unknown argument: $1"
        ;;
    esac
  done
}

parse_ensemble_range() {
  if [[ ! "${ENSEMBLE_RANGE}" =~ ^([0-9]+):([0-9]+)$ ]]; then
    die "--ensemble-range must look like START:END, for example 0:9"
  fi

  START_ENSEMBLE="${BASH_REMATCH[1]}"
  MAX_ENSEMBLE="${BASH_REMATCH[2]}"
  (( START_ENSEMBLE <= MAX_ENSEMBLE )) || die "--ensemble-range start must be <= end"
}

validate_args() {
  if (( SETUP_ONLY )); then
    (( ! SKIP_SETUP && ! TUI_MODE && ! DRY_RUN && ! GROUP_ABLATION_ONLY )) && [[ -z "${EXPERIMENT}${PLAN_PATH}" ]] || die "Use --setup on its own"
    return 0
  fi
  if (( SKIP_SETUP )) && [[ -z "${EXPERIMENT}${PLAN_PATH}" ]]; then
    TUI_MODE=1
  fi
  if (( GROUP_ABLATION_ONLY )) && (( TUI_MODE )); then
    die "--group-ablation-only is available only with --experiment and --layers"
  fi
  if (( GROUP_ABLATION_ONLY )) && [[ -n "${PLAN_PATH}" ]]; then
    die "--group-ablation-only cannot be combined with --plan"
  fi

  if (( TUI_MODE )); then
    return 0
  fi

  if [[ -n "${PLAN_PATH}" ]]; then
    assert_file "${PLAN_PATH}"
    return 0
  fi

  [[ -n "${EXPERIMENT}" ]] || die "Must provide --experiment"
  parse_experiment_metadata "${EXPERIMENT}" || die "Unsupported experiment: ${EXPERIMENT}"
  [[ -n "${LAYERS_SPEC}" ]] || die "--layers is required for experiment ${EXPERIMENT}"

  [[ "${NUM_GPUS}" =~ ^[1-9][0-9]*$ ]] || die "--num-gpus must be a positive integer"
  parse_ensemble_range
}

normalize_gpu_devices() {
  local selected
  selected="$("${PYTHON_BIN}" "${DEVICE_HELPER}" --select "${DEVICE_SPEC}" --count "${NUM_GPUS}")" || exit $?
  read -r -a GPU_DEVICES <<< "${selected}"
  if [[ "${selected}" == mps ]]; then
    # Ten independent seed workers share the same Apple GPU.
    GPU_DEVICES=(mps mps mps mps mps mps mps mps mps mps)
  fi
  (( ${#GPU_DEVICES[@]} > 0 )) || die "No compute device is available"
  ACTIVE_PIDS=()
  ACTIVE_LABELS=()
}

assert_runtime_inputs() {
  assert_file "${MAIN_PY}"

  if (( DRY_RUN )); then
    return 0
  fi

  if (( GROUP_ABLATION_ONLY )); then
    assert_file "${PROJECT_ROOT}/source_data/cluster_labels.csv"
  else
    assert_file "${PROJECT_ROOT}/source_data/datashare_with_return.pkl"
    assert_file "${PROJECT_ROOT}/source_data/PredictorData2024.xlsx"
    assert_file "${PROJECT_ROOT}/source_data/F-F_Research_Data_5_Factors_2x3.csv"
  fi

  local clean_cache_missing=0
  local cache_file
  for cache_file in \
    "${PROJECT_ROOT}/tmp/clean_X.pkl" \
    "${PROJECT_ROOT}/tmp/clean_y.pkl" \
    "${PROJECT_ROOT}/tmp/clean_mvel1.pkl" \
    "${PROJECT_ROOT}/tmp/clean_ff5.pkl"
  do
    if [[ ! -e "${cache_file}" ]]; then
      clean_cache_missing=1
      break
    fi
  done

  if (( clean_cache_missing )); then
    die "Missing tmp/clean_*.pkl. Run ./run.sh or python3 -m utils.prepare_data first."
  fi
}

detect_presplit_cache() {
  local first_tensor=""
  first_tensor="$(find "${PROJECT_ROOT}/tmp" -mindepth 2 -maxdepth 2 -name 'X_train_tensor.pt' -print -quit 2>/dev/null || true)"
  if [[ -n "${first_tensor}" ]]; then
    USE_PRESPLIT=1
  else
    USE_PRESPLIT=0
  fi
}

supports_live_progress() {
  [[ -t 1 ]] && [[ "${TERM:-dumb}" != "dumb" ]] && [[ -f "${PROGRESS_BAR_PY}" ]]
}

progress_bar_cmd() {
  local -a cmd=("${PYTHON_BIN}" "${PROGRESS_BAR_PY}" "$@")
  if [[ -n "${PROGRESS_ROOT}" ]]; then
    cmd+=(--root "${PROGRESS_ROOT}")
  fi
  "${cmd[@]}"
}

devices_csv() {
  local joined=""
  local device

  for device in "${GPU_DEVICES[@]}"; do
    if [[ -n "${joined}" ]]; then
      joined+=","
    fi
    joined+="${device}"
  done

  printf '%s' "${joined}"
}

progress_title() {
  local layers_text="${LAYERS_SPEC:-n/a}"
  printf 'run.sh %s | layers %s | ens %s:%s | devices %s' \
    "${EXPERIMENT}" \
    "${layers_text}" \
    "${START_ENSEMBLE}" \
    "${MAX_ENSEMBLE}" \
    "$(devices_csv)"
}

progress_event() {
  local message="$1"

  if (( ! PROGRESS_ENABLED )) || [[ -z "${message}" ]]; then
    return 0
  fi

  progress_bar_cmd event --session "${PROGRESS_SESSION_ID}" --message "${message}"
}

init_progress_session() {
  if (( DRY_RUN )) || ! supports_live_progress; then
    PROGRESS_ENABLED=0
    return 0
  fi

  PROGRESS_SESSION_ID="run_$(date '+%Y%m%d_%H%M%S')_$$"
  progress_bar_cmd \
    init \
    --session "${PROGRESS_SESSION_ID}" \
    --rows "${#GPU_DEVICES[@]}" \
    --title "$(progress_title)" \
    --footer-lines "${PROGRESS_FOOTER_LINES}"
  PROGRESS_ENABLED=1
  progress_event "[plan] experiment=${EXPERIMENT} layers=${LAYERS_SPEC:-n/a} ensembles=${START_ENSEMBLE}:${MAX_ENSEMBLE} devices=$(devices_csv)"
}

finalize_progress_session() {
  local status="$1"
  local message

  if (( ! PROGRESS_ENABLED )); then
    return 0
  fi

  if (( status == 0 )); then
    message="run.sh finished."
  else
    message="run.sh aborted with exit ${status}."
  fi

  progress_bar_cmd finalize --session "${PROGRESS_SESSION_ID}" --message "${message}" || true
  PROGRESS_ENABLED=0
}

pid_is_active() {
  local pid="$1"
  local stat

  stat="$(ps -p "${pid}" -o stat= 2>/dev/null | tr -d '[:space:]' || true)"
  [[ -n "${stat}" ]] || return 1
  [[ "${stat}" != Z* ]]
}

has_active_jobs() {
  local slot

  for ((slot = 0; slot < ${#GPU_DEVICES[@]}; slot += 1)); do
    if [[ -n "${ACTIVE_PIDS[$slot]:-}" ]]; then
      return 0
    fi
  done

  return 1
}

kill_active_jobs() {
  local slot
  local pid

  for ((slot = 0; slot < ${#GPU_DEVICES[@]}; slot += 1)); do
    pid="${ACTIVE_PIDS[$slot]:-}"
    [[ -n "${pid}" ]] || continue
    kill "${pid}" 2>/dev/null || true
  done
}

clear_slot() {
  local slot="$1"
  ACTIVE_PIDS[$slot]=""
  ACTIVE_LABELS[$slot]=""
}

reap_completed_jobs() {
  local slot
  local pid
  local status

  for ((slot = 0; slot < ${#GPU_DEVICES[@]}; slot += 1)); do
    pid="${ACTIVE_PIDS[$slot]:-}"
    [[ -n "${pid}" ]] || continue

    if pid_is_active "${pid}"; then
      continue
    fi

    if wait "${pid}"; then
      clear_slot "${slot}"
      continue
    else
      status=$?
    fi
    progress_event "[error] device ${GPU_DEVICES[$slot]} ${ACTIVE_LABELS[$slot]:-job} exited with status ${status}"
    clear_slot "${slot}"
    return "${status}"
  done

  return 0
}

find_idle_slot() {
  local slot

  for ((slot = 0; slot < ${#GPU_DEVICES[@]}; slot += 1)); do
    if [[ -z "${ACTIVE_PIDS[$slot]:-}" ]]; then
      printf '%s\n' "${slot}"
      return 0
    fi
  done

  return 1
}

wait_for_available_slot() {
  local slot

  while true; do
    if slot="$(find_idle_slot)"; then
      AVAILABLE_SLOT="${slot}"
      return 0
    fi

    reap_completed_jobs || return $?

    if slot="$(find_idle_slot)"; then
      AVAILABLE_SLOT="${slot}"
      return 0
    fi

    sleep 0.5
  done
}

wait_for_all_jobs() {
  while has_active_jobs; do
    reap_completed_jobs || return $?
    if has_active_jobs; then
      sleep 0.5
    fi
  done
}

run_foreground() {
  local cmd=("$@")
  if (( DRY_RUN )); then
    printf "[dry-run] "
    printf '%q ' "${cmd[@]}"
    printf '\n'
    return 0
  fi
  local -a env_args=()
  if (( PROGRESS_ENABLED )); then
    env_args+=("ResAssetPricing_PROGRESS_SESSION=${PROGRESS_SESSION_ID}")
    if [[ -n "${PROGRESS_ROOT}" ]]; then
      env_args+=("ResAssetPricing_PROGRESS_ROOT=${PROGRESS_ROOT}")
    fi
  fi
  env "${env_args[@]}" "${cmd[@]}"
}

expand_layers() {
  local spec="$1"

  "${PYTHON_BIN}" - "$spec" <<'PY'
import re
import sys

spec = sys.argv[1].strip()
if not spec:
    raise SystemExit("")

parts = [part.strip() for part in spec.split(",")]
layers = []

for part in parts:
    if not part:
        continue
    if ":" not in part:
        if not re.fullmatch(r"-?[0-9]+", part):
            raise SystemExit(f"Invalid layer token: {part}")
        layers.append(int(part))
        continue

    seg = part.split(":")
    if len(seg) not in (2, 3):
        raise SystemExit(f"Invalid layer token: {part}")

    start = int(seg[0])
    end = int(seg[1])
    step = int(seg[2]) if len(seg) == 3 else 1
    if step <= 0:
        raise SystemExit(f"Invalid step in layer token: {part}")

    if start <= end:
        rng = range(start, end + 1, step)
    else:
        rng = range(start, end - 1, -step)
    layers.extend(rng)

print(" ".join(str(v) for v in layers))
PY
}

validate_layer_for_experiment() {
  local experiment="$1"
  local layer="$2"
  parse_experiment_metadata "${experiment}" || die "Unsupported experiment: ${experiment}"
  (( layer >= PARSED_EXPERIMENT_SEED && layer <= MAX_HIDDEN_LAYER_COUNT )) || die \
    "Layer ${layer} is invalid for ${experiment}; expected ${PARSED_EXPERIMENT_SEED}:${MAX_HIDDEN_LAYER_COUNT}"
}

queue_job() {
  local label="$1"
  shift
  local -a cmd=("$@")
  local slot
  local gpu
  local -a env_args=()

  if (( DRY_RUN )); then
    slot="${DRY_RUN_NEXT_SLOT}"
    DRY_RUN_NEXT_SLOT=$(((DRY_RUN_NEXT_SLOT + 1) % ${#GPU_DEVICES[@]}))
  else
    wait_for_available_slot || return $?
    slot="${AVAILABLE_SLOT}"
  fi
  gpu="${GPU_DEVICES[$slot]}"
  case "${gpu}" in
    cpu|mps)
      env_args=("CUDA_VISIBLE_DEVICES=")
      cmd+=(--device "${gpu}")
      ;;
    *)
      env_args=("CUDA_VISIBLE_DEVICES=${gpu}")
      cmd+=(--device cuda:0)
      ;;
  esac
  if (( DRY_RUN )); then
    printf '[dry-run] '
    printf '%q ' "${env_args[@]}" "${cmd[@]}"
    printf '\n'
    return 0
  fi

  if (( PROGRESS_ENABLED )); then
    env_args+=("ResAssetPricing_PROGRESS_SESSION=${PROGRESS_SESSION_ID}")
    env_args+=("ResAssetPricing_PROGRESS_SLOT=${slot}")
    env_args+=("ResAssetPricing_PROGRESS_ROW_LABEL=device ${gpu}")
    env_args+=("ResAssetPricing_PROGRESS_LABEL=${label}")
    if [[ -n "${PROGRESS_ROOT}" ]]; then
      env_args+=("ResAssetPricing_PROGRESS_ROOT=${PROGRESS_ROOT}")
    fi
  fi

  env "${env_args[@]}" "${cmd[@]}" &
  ACTIVE_PIDS[$slot]="$!"
  ACTIVE_LABELS[$slot]="${label}"
}

queue_all_done() {
  if has_active_jobs; then
    wait_for_all_jobs
  fi
}

config_for_experiment() {
  local experiment="$1"
  local layer="$2"
  parse_experiment_metadata "${experiment}" || die "Unknown experiment: ${experiment}"

  case "${PARSED_EXPERIMENT_KIND}" in
    nn)
      printf 'NNS%sD%sConfig' "${PARSED_EXPERIMENT_SEED}" "${layer}"
      ;;
    nnp)
      printf 'NNpS%sD%sConfig' "${PARSED_EXPERIMENT_SEED}" "${layer}"
      ;;
    resnet)
      printf 'ResNetS%sD%sConfig' "${PARSED_EXPERIMENT_SEED}" "${layer}"
      ;;
    resnetp)
      printf 'ResNetpS%sD%sConfig' "${PARSED_EXPERIMENT_SEED}" "${layer}"
      ;;
    *)
      die "Unknown experiment: ${experiment}"
      ;;
  esac
}

asset_prefix_for_experiment() {
  local experiment="$1"
  local layer="$2"
  parse_experiment_metadata "${experiment}" || die "Unknown experiment: ${experiment}"

  case "${PARSED_EXPERIMENT_KIND}" in
    nn)
      printf 'nns%sd%s' "${PARSED_EXPERIMENT_SEED}" "${layer}"
      ;;
    nnp)
      printf 'nnps%sd%s' "${PARSED_EXPERIMENT_SEED}" "${layer}"
      ;;
    resnet)
      printf 'resnets%sd%s' "${PARSED_EXPERIMENT_SEED}" "${layer}"
      ;;
    resnetp)
      printf 'resnetps%sd%s' "${PARSED_EXPERIMENT_SEED}" "${layer}"
      ;;
    *)
      die "Cannot derive asset_prefix for experiment: ${experiment}"
      ;;
  esac
}

job_label_for_launch() {
  local asset_prefix="$1"
  local ensemble="$2"
  local suffix="${3:-}"
  local label

  label="$(printf '%s e%02d' "${asset_prefix}" "${ensemble}")"
  if [[ -n "${suffix}" ]]; then
    label="${label} ${suffix}"
  fi
  printf '%s' "${label}"
}

queue_ensembles_for_config() {
  local config_cls="$1"
  local asset_prefix="$2"
  shift 2
  local -a extra_args=("$@")
  local ensemble

  for ensemble in $(seq "${START_ENSEMBLE}" "${MAX_ENSEMBLE}"); do
    local -a cmd=(
      "${PYTHON_BIN}" "${MAIN_PY}"
      --config_cls "${config_cls}"
      --asset_prefix "${asset_prefix}"
      --train
      --predict
      --ensemble "${ensemble}"
    )
    if (( USE_PRESPLIT )); then
      cmd+=(--use_pre_split_dataset)
    fi
    cmd+=("${extra_args[@]}")
    queue_job "$(job_label_for_launch "${asset_prefix}" "${ensemble}")" "${cmd[@]}"
  done

  queue_all_done
}

run_econ_for_config() {
  local config_cls="$1"
  local asset_prefix="$2"
  shift 2
  local -a extra_args=("$@")
  local -a econ_cmd=(
    "${PYTHON_BIN}" "${MAIN_PY}"
    --config_cls "${config_cls}"
    --asset_prefix "${asset_prefix}"
    --econ
    --device cpu
    --start_ensemble "${START_ENSEMBLE}"
    --max_ensemble "${MAX_ENSEMBLE}"
    --econ_mkt_cap "${REPORT_GROUPS[@]}"
    "${ECON_WEIGHT_ARGS[@]}"
  )
  econ_cmd+=("${extra_args[@]}")
  run_foreground "${econ_cmd[@]}"
}

run_group_ablation_for_config() {
  local config_cls="$1"
  local asset_prefix="$2"
  shift 2
  local -a extra_args=("$@")
  local -a ablation_cmd=(
    "${PYTHON_BIN}" "${MAIN_PY}"
    --config_cls "${config_cls}"
    --asset_prefix "${asset_prefix}"
    --group_ablation
    --use_pre_split_dataset
    --start_ensemble "${START_ENSEMBLE}"
    --max_ensemble "${MAX_ENSEMBLE}"
    --econ_mkt_cap ALL
    "${ECON_WEIGHT_ARGS[@]}"
  )
  ablation_cmd+=("${extra_args[@]}")
  queue_job "$(job_label_for_launch "${asset_prefix}" "${START_ENSEMBLE}" "group-ablation")" "${ablation_cmd[@]}"
}

run_config_with_econ() {
  local config_cls="$1"
  local asset_prefix="$2"
  shift 2
  local -a extra_args=("$@")

  progress_event "[stage] ${asset_prefix} train+predict"
  queue_ensembles_for_config "${config_cls}" "${asset_prefix}" "${extra_args[@]}"
  if (( ! RUN_ECON )); then
    progress_event "[skip] ${asset_prefix} econ disabled"
    return 0
  fi
  progress_event "[stage] ${asset_prefix} econ"
  run_econ_for_config "${config_cls}" "${asset_prefix}" "${extra_args[@]}"
}

dispatch_layer_job() {
  local experiment="$1"
  local layer="$2"
  local config_cls
  local asset_prefix
  local -a extra_args=()

  config_cls="$(config_for_experiment "${experiment}" "${layer}")"
  asset_prefix="$(asset_prefix_for_experiment "${experiment}" "${layer}")"
  parse_experiment_metadata "${experiment}" || die "Unknown experiment: ${experiment}"

  case "${PARSED_EXPERIMENT_KIND}" in
    nn|resnet)
      extra_args=("${PLAIN_RESNET_NN_TRAINING_ARGS[@]}")
      ;;
    nnp|resnetp)
      extra_args=("${COMMON_TRAINING_ARGS[@]}")
      ;;
    *)
      extra_args=()
      ;;
  esac

  if (( GROUP_ABLATION_ONLY )); then
    progress_event "[stage] ${asset_prefix} group-ablation: econ and stock predictions (no training)"
    run_group_ablation_for_config "${config_cls}" "${asset_prefix}" "${extra_args[@]}"
  else
    run_config_with_econ "${config_cls}" "${asset_prefix}" "${extra_args[@]}"
  fi
}

run_experiment() {
  local -a expanded=()
  local layer

  read -r -a expanded <<< "$(expand_layers "${LAYERS_SPEC}")"
  if (( ${#expanded[@]} == 0 )); then
    die "No valid layers resolved from: ${LAYERS_SPEC}"
  fi

  if (( GROUP_ABLATION_ONLY )); then
    for layer in "${expanded[@]}"; do
      validate_layer_for_experiment "${EXPERIMENT}" "${layer}"
    done
    parse_experiment_metadata "${EXPERIMENT}" || die "Unknown experiment: ${EXPERIMENT}"
    local anchor_layer="${PARSED_EXPERIMENT_SEED}"
    local -a ablation_layers=("${anchor_layer}")
    for layer in "${expanded[@]}"; do
      [[ "${layer}" == "${anchor_layer}" ]] || ablation_layers+=("${layer}")
    done
    expanded=("${ablation_layers[@]}")
    progress_event "[stage] ${EXPERIMENT} includes shallow-anchor depth ${anchor_layer}"
  fi

  for layer in "${expanded[@]}"; do
    validate_layer_for_experiment "${EXPERIMENT}" "${layer}"
    dispatch_layer_job "${EXPERIMENT}" "${layer}"
  done
}

on_exit() {
  local status="$1"

  if (( status != 0 )) && (( ! TUI_MODE )) && [[ -z "${PLAN_PATH}" ]]; then
    kill_active_jobs || true
  fi
  if (( ! TUI_MODE )) && [[ -z "${PLAN_PATH}" ]]; then
    finalize_progress_session "${status}"
  fi
}

main() {
  parse_args "$@"
  validate_args

  source "${PROJECT_ROOT}/utils/bootstrap_environment.sh"
  bootstrap_detect_platform
  select_python
  if (( ! SKIP_SETUP )); then
    bootstrap_environment
  fi
  command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "Python is unavailable. Run ./run.sh --setup."
  "${PYTHON_BIN}" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || die "Python 3.10+ is required; the reference version is 3.13.5. Run ./run.sh --setup."
  if (( SETUP_ONLY )); then
    printf 'Environment ready. Run ./run.sh to open the launcher.\n'
    return 0
  fi
  assert_file "${DEVICE_HELPER}"

  if (( ! DRY_RUN )) && [[ -t 0 && -t 1 && ! -f "${PROJECT_ROOT}/source_data/datashare_with_return.pkl" && -f "${PROJECT_ROOT}/source_data/t9ftbki1xauaox25.sas7bdat" ]]; then
    local convert_reply
    printf 'Found SAS source. Convert it to datashare_with_return.pkl now? [y/N] '
    IFS= read -r convert_reply || convert_reply=n
    case "${convert_reply}" in
      y|Y|yes|YES)
        (cd "${PROJECT_ROOT}" && "${PYTHON_BIN}" utils/sas7bdat2pickle.py)
        # The TUI prepares clean inputs when a run starts; CLI runs need them now.
        if (( ! TUI_MODE )) && [[ -z "${PLAN_PATH}" ]]; then
          "${PYTHON_BIN}" "${PROJECT_ROOT}/utils/prepare_data.py" --project-root "${PROJECT_ROOT}"
        fi
        ;;
    esac
  fi

  if (( TUI_MODE )); then
    assert_file "${RUN_TUI_PY}"
    local -a tui_cmd=("${PYTHON_BIN}" "${RUN_TUI_PY}" interactive --project-root "${PROJECT_ROOT}" --python-bin "${PYTHON_BIN}")
    if (( DRY_RUN )); then
      tui_cmd+=(--dry-run)
    fi
    "${tui_cmd[@]}"
    return 0
  fi

  if [[ -n "${PLAN_PATH}" ]]; then
    assert_file "${RUN_TUI_PY}"
    local -a plan_cmd=("${PYTHON_BIN}" "${RUN_TUI_PY}" execute --plan "${PLAN_PATH}" --project-root "${PROJECT_ROOT}" --python-bin "${PYTHON_BIN}")
    if (( DRY_RUN )); then
      plan_cmd+=(--dry-run)
    fi
    if (( ! RUN_ECON )); then
      plan_cmd+=(--no-econ)
    fi
    "${plan_cmd[@]}"
    return 0
  fi

  normalize_gpu_devices
  assert_runtime_inputs
  detect_presplit_cache
  if (( GROUP_ABLATION_ONLY )) && (( ! DRY_RUN )) && (( ! USE_PRESPLIT )); then
    die "--group-ablation-only requires complete pre-split test tensors under tmp/YYYYMMDD"
  fi
  init_progress_session
  run_experiment
  if (( GROUP_ABLATION_ONLY )); then
    queue_all_done
  fi
}

trap 'on_exit $?' EXIT

main "$@"
