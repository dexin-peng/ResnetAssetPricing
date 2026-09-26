#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
try:
    import curses
except ImportError as exc:
    raise SystemExit("Terminal support (curses) is missing. Use WSL2 on Windows, or install windows-curses; on Linux/macOS use a Python build with curses.") from exc
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import zipfile
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.progress_bar import append_event, ensure_row_templates, finalize_session, init_session, update_row_state
from utils.runtime_device import (available_devices, worker_device, worker_environment,
                                  default_slots, device_slots, device_message)


CHECK_MARK = "√"
ASCII_CHECK_MARK = "v"
ASCII_TUI_ENV = "ResAssetPricing_ASCII_TUI"

COMMON_TRAINING_ARGS = (
    "--epochs",
    "100",
    "--batch_size",
    "10000",
    "--early_stopping_patience",
    "5",
    "--l2",
    "--weight_decay",
    "1e-4",
    "--initial_lr",
    "1e-4",
)
PLAIN_RESNET_NN_TRAINING_ARGS = (
    "--epochs",
    "100",
    "--batch_size",
    "10000",
    "--early_stopping_patience",
    "5",
    "--l1",
    "--l1_lambda",
    "1e-4",
    "--initial_lr",
    "1e-4",
)
REPORT_GROUPS = ("ALL", "TOP80%", "BOTTOM80%")
ECON_WEIGHT_ARGS = ("--econ_weight_type", "vw", "ew")
PUBLIC_FAMILY_STEMS = ("nns", "nnps", "resnets", "resnetps")
GROUP_ABLATION_KINDS = {"nn", "nnp", "resnet", "resnetp"}
GROUP_ABLATION_GPU_CONCURRENCY = 8
DEFAULT_ENSEMBLE_SPEC = "0:9"
GPU_UTIL_REFRESH_SECONDS = 1.0
UI_POLL_INTERVAL_MS = 200
MAX_HIDDEN_LAYER_COUNT = 20
PUBLIC_WIDTH_SCHEDULES = range(1, 5)


@dataclass(frozen=True)
class ExperimentDef:
    name: str
    role: str
    valid_layers: str
    default_layers: str
    default_enabled: bool
    default_blocking_batch: int


def parse_seeded_experiment_name(experiment: str) -> tuple[str, int]:
    match = re.fullmatch(r"nns([1-4])", experiment)
    if match:
        return "nn", int(match.group(1))

    match = re.fullmatch(r"nnps([1-4])", experiment)
    if match:
        return "nnp", int(match.group(1))

    match = re.fullmatch(r"resnets([1-4])", experiment)
    if match:
        return "resnet", int(match.group(1))

    match = re.fullmatch(r"resnetps([1-4])", experiment)
    if match:
        return "resnetp", int(match.group(1))

    raise ValueError(f"Unsupported experiment: {experiment}")


def _build_seeded_experiment_registry() -> tuple[list[str], dict[str, ExperimentDef]]:
    experiments: dict[str, ExperimentDef] = {}
    order: list[str] = []
    default_blocking_batch = 1
    family_specs = (
        ("resnetps", "ResNet+"),
        ("nnps", "NN+"),
        ("resnets", "ResNet"),
        ("nns", "NN"),
    )

    for prefix, role in family_specs:
        for seed in PUBLIC_WIDTH_SCHEDULES:
            name = f"{prefix}{seed}"
            order.append(name)
            experiments[name] = ExperimentDef(
                name=name,
                role=f"{role} (s{seed})",
                valid_layers=f"{seed}:{MAX_HIDDEN_LAYER_COUNT}",
                default_layers=f"{seed}:{MAX_HIDDEN_LAYER_COUNT}",
                default_enabled=True,
                default_blocking_batch=default_blocking_batch,
            )
            default_blocking_batch += 1

    return order, experiments


EXPERIMENT_ORDER, EXPERIMENTS = _build_seeded_experiment_registry()


@dataclass
class CellState:
    enabled: bool
    blocking_batch: int
    gpus: list[str]


@dataclass
class LayerState:
    cells: dict[int, CellState]


@dataclass
class ExperimentState:
    layers_spec: str
    layer_states: dict[int, LayerState]


@dataclass
class ScheduledGpuJob:
    experiment_index: int
    blocking_batch: int
    layer: int
    assigned_gpu: str
    command: list[str]
    label: str
    progress_group: str
    progress_item: str
    completion_key: tuple[int, ...]
    order_key: tuple[int, ...]


@dataclass
class ScheduledForegroundJob:
    experiment_index: int
    blocking_batch: int
    command: list[str]
    label: str
    depends_on: tuple[int, ...]
    order_key: tuple[int, ...]


class AsyncForegroundJobManager:
    def __init__(
        self,
        *,
        project_root: Path,
        session_id: str | None,
        dry_run: bool,
    ) -> None:
        self.project_root = project_root
        self.session_id = session_id
        self.dry_run = dry_run
        self.active: list[tuple[subprocess.Popen[str], ScheduledForegroundJob]] = []

    def launch_jobs(self, jobs: list[ScheduledForegroundJob]) -> None:
        if not jobs:
            return

        for job in jobs:
            env: dict[str, str] | None = None
            if self.session_id and supports_live_progress():
                env = {
                    "ResAssetPricing_PROGRESS_SESSION": self.session_id,
                    "ResAssetPricing_PROGRESS_GROUPED": "1",
                }
            elif not self.dry_run:
                print(f"[econ] {job.label}")

            if self.dry_run:
                run_subprocess(job.command, cwd=self.project_root, env=env, dry_run=True)
                continue

            merged_env = os.environ.copy()
            if env:
                merged_env.update(env)
            process = subprocess.Popen(job.command, cwd=str(self.project_root), env=merged_env, text=True)
            self.active.append((process, job))

    def poll(self) -> None:
        if self.dry_run or not self.active:
            return

        failure = self._collect_finished(wait=False)
        if failure is None:
            return

        self.terminate_all()
        self._collect_finished(wait=True)
        raise failure

    def wait(self) -> None:
        if self.dry_run or not self.active:
            return

        failure = self._collect_finished(wait=True)
        if failure is not None:
            raise failure

    def terminate_all(self) -> None:
        for process, _ in self.active:
            if process.poll() is None:
                process.terminate()

    def reap_all(self) -> None:
        if self.dry_run or not self.active:
            return
        self._collect_finished(wait=True)

    def _collect_finished(self, *, wait: bool) -> subprocess.CalledProcessError | None:
        failure: subprocess.CalledProcessError | None = None
        while self.active:
            made_progress = False
            for index in range(len(self.active) - 1, -1, -1):
                process, job = self.active[index]
                status = process.wait() if wait else process.poll()
                if status is None:
                    continue
                made_progress = True
                self.active.pop(index)
                if status != 0 and failure is None:
                    failure = subprocess.CalledProcessError(status, job.command)
            if failure is not None or not wait or not self.active:
                return failure
            if not made_progress:
                time.sleep(0.2)
        return failure


def parse_integer_spec(spec: str, label: str, *, allow_desc: bool = True) -> list[int]:
    text = str(spec or "").strip()
    if not text:
        return []

    values: list[int] = []
    seen: set[int] = set()
    for chunk in text.split(","):
        part = chunk.strip()
        if not part:
            continue
        if ":" not in part:
            value = int(part)
            if value not in seen:
                seen.add(value)
                values.append(value)
            continue

        segments = [segment.strip() for segment in part.split(":")]
        if len(segments) not in (2, 3):
            raise ValueError(f"Invalid {label} token: {part}")
        start = int(segments[0])
        end = int(segments[1])
        step = int(segments[2]) if len(segments) == 3 else 1
        if step <= 0:
            raise ValueError(f"Invalid {label} step: {part}")

        if start <= end:
            rng = range(start, end + 1, step)
        else:
            if not allow_desc:
                raise ValueError(f"Descending {label} ranges are not supported: {part}")
            rng = range(start, end - 1, -step)

        for value in rng:
            if value not in seen:
                seen.add(value)
                values.append(value)

    return values


def parse_layer_range_shortcut(spec: str) -> tuple[int, int]:
    text = str(spec or "").strip().lower()
    if not text:
        raise ValueError("Layer range shortcut cannot be empty")
    text = re.sub(r"\s+", "", text)
    for suffix in ("layers", "layer", "depths", "depth", "层"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            break
    text = text.replace("到", "-").replace("至", "-").replace("—", "-").replace("–", "-")
    if ":" in text and "-" not in text:
        parts = text.split(":")
    else:
        parts = text.split("-")
    if len(parts) == 1 and parts[0]:
        start = end = int(parts[0])
    elif len(parts) == 2 and parts[0] and parts[1]:
        start, end = int(parts[0]), int(parts[1])
    else:
        raise ValueError(f"Invalid layer range shortcut: {spec}")
    if start <= 0 or end <= 0:
        raise ValueError("Layer range shortcut must use positive depths")
    if start > end:
        raise ValueError(f"Descending layer range shortcut is not supported: {spec}")
    return start, end


def seeded_family_members(experiment: str) -> list[str]:
    family_kind, _ = parse_seeded_experiment_name(experiment)
    members = []
    for name in EXPERIMENT_ORDER:
        candidate_kind, _ = parse_seeded_experiment_name(name)
        if candidate_kind == family_kind:
            members.append(name)
    return members


def validate_layers(experiment: str, layers: list[int]) -> None:
    if not layers:
        raise ValueError(f"{experiment} has no selected layers")

    _, seed = parse_seeded_experiment_name(experiment)
    for layer in layers:
        if seed <= layer <= MAX_HIDDEN_LAYER_COUNT:
            continue
        raise ValueError(f"{experiment} layer {layer} is invalid; expected {seed}:{MAX_HIDDEN_LAYER_COUNT}")


def detect_devices() -> list[str]:
    return available_devices()


def detect_gpu_utils(devices: list[str]) -> dict[str, int | None]:
    if not devices or devices[0] in {"cpu", "mps"}:
        return {device: None for device in devices}

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={','.join(devices)}",
                "--query-gpu=index,uuid,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return {device: None for device in devices}

    by_key: dict[str, int | None] = {}
    for raw_line in result.stdout.splitlines():
        parts = [part.strip() for part in raw_line.split(",", 2)]
        if len(parts) != 3:
            continue
        index, uuid, util_text = parts
        try:
            util_value: int | None = int(round(float(util_text)))
        except ValueError:
            util_value = None
        by_key[index] = util_value
        by_key[uuid] = util_value

    return {device: by_key.get(device) for device in devices}


def parse_gpu_spec(spec: str, devices: list[str]) -> list[str]:
    raw = str(spec or "").strip()
    if not raw:
        raise ValueError("Device selection cannot be empty")
    if not devices:
        raise ValueError("No compute devices are available")
    if raw.lower() in {"all", "*", "auto"}:
        return list(devices)

    selected: list[str] = []
    seen: set[str] = set()
    if ":" in raw and all(token.strip().isdigit() for token in raw.replace(":", ",").split(",")):
        positions = parse_integer_spec(raw, "gpu", allow_desc=False)
        for position in positions:
            if position < 0 or position >= len(devices):
                raise ValueError(f"Device position {position} is out of range")
            device = devices[position]
            if device not in seen:
                seen.add(device)
                selected.append(device)
        return selected

    for chunk in raw.split(","):
        token = chunk.strip()
        if not token:
            continue
        if token.isdigit():
            position = int(token)
            if 0 <= position < len(devices):
                device = devices[position]
            elif token in devices:
                device = token
            else:
                raise ValueError(f"Unknown Device position or id: {token}")
        elif token in devices:
            device = token
        else:
            raise ValueError(f"Unknown device token: {token}")

        if device not in seen:
            seen.add(device)
            selected.append(device)

    if not selected:
        raise ValueError("Device selection resolves to an empty set")
    return selected


def parse_gpu_slot_spec(spec: str, devices: list[str], current: dict[str, int]) -> dict[str, int]:
    raw = str(spec or "").strip()
    if not raw:
        raise ValueError("Device slot spec cannot be empty")
    if not devices:
        raise ValueError("No compute devices are available")

    if raw.isdigit():
        slots = int(raw)
        if slots <= 0:
            raise ValueError("Device slot count must be >= 1")
        return {device: slots for device in devices}

    updated = {device: max(int(current.get(device, 1)), 1) for device in devices}
    for chunk in raw.split(","):
        token = chunk.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError("Use a positive integer for all GPUs or entries like 0=2,1=1")
        lhs, rhs = token.split("=", 1)
        lhs = lhs.strip()
        rhs = rhs.strip()
        if not rhs.isdigit() or int(rhs) <= 0:
            raise ValueError(f"Invalid Device slot count: {token}")
        slot_count = int(rhs)

        targets: list[str]
        if lhs.lower() == "all":
            targets = list(devices)
        elif lhs in devices:
            targets = [lhs]
        elif lhs.isdigit():
            position = int(lhs)
            if 0 <= position < len(devices):
                targets = [devices[position]]
            else:
                raise ValueError(f"Device position {position} is out of range")
        else:
            raise ValueError(f"Unknown device token in slot spec: {lhs}")

        for device in targets:
            updated[device] = slot_count

    return updated


def device_positions(selected: list[str], devices: list[str]) -> str:
    if not selected:
        return "-"
    if set(selected) == set(devices):
        return "all"
    positions: list[str] = []
    for device in selected:
        if device in devices:
            positions.append(str(devices.index(device)))
        else:
            positions.append(str(device))
    return ",".join(positions)


def available_gpu_positions(devices: list[str]) -> str:
    if not devices:
        return "none"
    return ",".join(str(index) for index, _ in enumerate(devices))


def gpu_slots_summary(gpu_slots: dict[str, int], devices: list[str]) -> str:
    if not devices:
        return "none"
    return ",".join(f"{index}={max(int(gpu_slots.get(device, 20)), 1)}" for index, device in enumerate(devices))


def compact_integer_spec(values: list[int]) -> str:
    if not values:
        return "n/a"

    ordered = [int(value) for value in values]
    parts: list[str] = []
    start = ordered[0]
    end = ordered[0]

    for value in ordered[1:]:
        if value == end + 1:
            end = value
            continue
        parts.append(f"{start}:{end}" if start != end else str(start))
        start = value
        end = value

    parts.append(f"{start}:{end}" if start != end else str(start))
    return ",".join(parts)


def selected_ensemble_range(selected_ensembles: list[int], *, experiment: str, layer: int) -> tuple[int, int]:
    ordered = sorted({int(ensemble) for ensemble in selected_ensembles})
    if not ordered:
        raise ValueError(f"{experiment} layer {layer} has no selected ensembles")

    expected = list(range(ordered[0], ordered[-1] + 1))
    if ordered != expected:
        raise ValueError(
            f"{experiment} layer {layer} selected ensembles must be contiguous for econ execution"
        )

    return ordered[0], ordered[-1]


def normalized_gpu_slots(gpu_slots: dict[str, int] | None, devices: list[str]) -> dict[str, int]:
    raw = gpu_slots or {}
    return {device: device_slots(device, raw.get(device, default_slots(device))) for device in devices}


def depth_concurrency_limit(layer: int, device: str | None = None) -> int:
    if device == "mps":
        return default_slots(device)
    if layer <= 12:
        return 10
    if layer <= 20:
        return 5
    if layer < 28:
        return 3
    return 2


def layer_load_weight(layer: int) -> float:
    return 20.0 / float(depth_concurrency_limit(layer))


def effective_gpu_total_limit(
    gpu: str,
    *,
    candidate_layer: int,
    gpu_limits: dict[str, int],
    active_layer_limits: dict[str, list[int]],
) -> int:
    if gpu == "mps":
        return device_slots(gpu, gpu_limits.get(gpu, default_slots(gpu)))
    limit = min(gpu_limits.get(gpu, 20), depth_concurrency_limit(candidate_layer))
    if active_layer_limits.get(gpu):
        limit = min(limit, min(active_layer_limits[gpu]))
    return max(limit, 1)


def balanced_gpu_assignment(ensemble_ids: list[int], devices: list[str]) -> dict[int, str]:
    if not devices:
        raise ValueError("No compute devices are available")
    total = len(ensemble_ids)
    device_count = len(devices)
    base, remainder = divmod(total, device_count)
    assignment: dict[int, str] = {}
    cursor = 0
    for device_index, device in enumerate(devices):
        take = base + (1 if device_index < remainder else 0)
        for ensemble in ensemble_ids[cursor : cursor + take]:
            assignment[ensemble] = device
        cursor += take
    if cursor < total:
        for ensemble in ensemble_ids[cursor:]:
            assignment[ensemble] = devices[-1]
    return assignment


def assign_counts_by_load(
    devices: list[str],
    *,
    batch_counts: dict[str, int],
    batch_caps: dict[str, int | None],
    batch_loads: dict[str, float],
    layer: int,
    total_cells: int,
    enforce_layer_cap: bool = True,
) -> dict[str, int] | None:
    if not devices:
        return None
    layer_cap = depth_concurrency_limit(layer, devices[0])
    counts = {device: 0 for device in devices}
    temp_counts = dict(batch_counts)
    temp_caps = dict(batch_caps)
    temp_loads = dict(batch_loads)
    weight = layer_load_weight(layer)

    for _ in range(total_cells):
        chosen_device: str | None = None
        chosen_key: tuple[float, int, int] | None = None
        for device_index, device in enumerate(devices):
            effective_cap: int | None = None
            if enforce_layer_cap:
                effective_cap = layer_cap if temp_caps.get(device) is None else min(int(temp_caps[device]), layer_cap)
            if effective_cap is not None and temp_counts.get(device, 0) >= effective_cap:
                continue
            candidate_key = (temp_loads.get(device, 0.0), temp_counts.get(device, 0), device_index)
            if chosen_key is None or candidate_key < chosen_key:
                chosen_key = candidate_key
                chosen_device = device
        if chosen_device is None:
            return None
        if enforce_layer_cap:
            temp_caps[chosen_device] = (
                layer_cap
                if temp_caps.get(chosen_device) is None
                else min(int(temp_caps[chosen_device]), layer_cap)
            )
        temp_counts[chosen_device] = temp_counts.get(chosen_device, 0) + 1
        temp_loads[chosen_device] = temp_loads.get(chosen_device, 0.0) + weight
        counts[chosen_device] += 1

    return counts


def assign_layer_chunk_to_devices(cells: list[CellState], counts: dict[str, int], blocking_batch: int) -> None:
    cursor = 0
    ordered_devices = [device for device, count in counts.items() if count > 0]
    ordered_cells = cells[:]
    for device in ordered_devices:
        take = counts[device]
        for cell in ordered_cells[cursor : cursor + take]:
            cell.gpus = [device]
            cell.blocking_batch = blocking_batch
        cursor += take


def auto_plan_blocking_batches_for_experiment(experiment: str, row: ExperimentState, ensemble_ids: list[int]) -> None:
    batch = 1
    batch_counts: dict[str, int] = defaultdict(int)
    batch_caps: dict[str, int | None] = {}

    def start_new_batch() -> None:
        nonlocal batch, batch_counts, batch_caps
        batch += 1
        batch_counts = defaultdict(int)
        batch_caps = {}

    for layer in resolved_layers_for_state(experiment, row):
        layer_state = row.layer_states[layer]
        layer_cells_by_gpu: dict[str, list[CellState]] = defaultdict(list)
        for ensemble in ensemble_ids:
            cell = layer_state.cells[ensemble]
            if not cell.enabled or len(cell.gpus) != 1:
                continue
            layer_cells_by_gpu[cell.gpus[0]].append(cell)

        if not layer_cells_by_gpu:
            continue
        layer_cap = depth_concurrency_limit(layer, "mps" if set(layer_cells_by_gpu) == {"mps"} else None)

        def fits_current_batch() -> bool:
            for gpu, cells in layer_cells_by_gpu.items():
                next_cap = layer_cap if batch_caps.get(gpu) is None else min(batch_caps[gpu], layer_cap)
                if batch_counts[gpu] + len(cells) > next_cap:
                    return False
            return True

        def fits_fresh_batch() -> bool:
            return all(len(cells) <= layer_cap for cells in layer_cells_by_gpu.values())

        if fits_current_batch():
            for gpu, cells in layer_cells_by_gpu.items():
                batch_caps[gpu] = layer_cap if batch_caps.get(gpu) is None else min(batch_caps[gpu], layer_cap)
                batch_counts[gpu] += len(cells)
                for cell in cells:
                    cell.blocking_batch = batch
            continue

        if batch_counts:
            start_new_batch()

        if fits_fresh_batch():
            for gpu, cells in layer_cells_by_gpu.items():
                batch_caps[gpu] = layer_cap
                batch_counts[gpu] += len(cells)
                for cell in cells:
                    cell.blocking_batch = batch
            continue

        chunk_count = max((len(cells) + layer_cap - 1) // layer_cap for cells in layer_cells_by_gpu.values())
        for chunk_index in range(chunk_count):
            if chunk_index > 0:
                start_new_batch()
            for gpu, cells in layer_cells_by_gpu.items():
                chunk = cells[chunk_index * layer_cap : (chunk_index + 1) * layer_cap]
                if not chunk:
                    continue
                batch_caps[gpu] = layer_cap
                batch_counts[gpu] += len(chunk)
                for cell in chunk:
                    cell.blocking_batch = batch


def safe_curs_set(value: int) -> None:
    try:
        curses.curs_set(value)
    except curses.error:
        pass


def safe_addnstr(window, y: int, x: int, text: str, maxlen: int, attr: int = 0) -> None:
    if maxlen <= 0 or y < 0 or x < 0:
        return
    try:
        window.addnstr(y, x, text, maxlen, attr)
    except UnicodeEncodeError:
        try:
            window.addnstr(y, x, text.replace(CHECK_MARK, ASCII_CHECK_MARK), maxlen, attr)
        except (curses.error, UnicodeEncodeError):
            pass
    except curses.error:
        pass


def should_use_ascii_tui_symbols(window) -> bool:
    forced = os.environ.get(ASCII_TUI_ENV)
    if forced is not None:
        normalized = forced.strip().lower()
        if normalized in {"1", "true", "yes", "on", "ascii"}:
            return True
        if normalized in {"0", "false", "no", "off", "unicode", "utf8", "utf-8"}:
            return False

    for encoding in (getattr(window, "encoding", None), sys.stdout.encoding):
        if not encoding:
            continue
        try:
            CHECK_MARK.encode(encoding)
        except (LookupError, UnicodeEncodeError):
            return True
        return False
    return False


def config_for_experiment(experiment: str, layer: int) -> str:
    family_kind, seed = parse_seeded_experiment_name(experiment)
    if family_kind == "nn":
        return f"NNS{seed}D{layer}Config"
    if family_kind == "nnp":
        return f"NNpS{seed}D{layer}Config"
    if family_kind == "resnet":
        return f"ResNetS{seed}D{layer}Config"
    if family_kind == "resnetp":
        return f"ResNetpS{seed}D{layer}Config"
    raise ValueError(f"Unsupported experiment: {experiment}")


def asset_prefix_for_experiment(experiment: str, layer: int) -> str:
    family_kind, seed = parse_seeded_experiment_name(experiment)
    if family_kind == "nn":
        return f"nns{seed}d{layer}"
    if family_kind == "nnp":
        return f"nnps{seed}d{layer}"
    if family_kind == "resnet":
        return f"resnets{seed}d{layer}"
    if family_kind == "resnetp":
        return f"resnetps{seed}d{layer}"
    raise ValueError(f"Unsupported experiment: {experiment}")


def extra_args_for_experiment(experiment: str) -> tuple[str, ...]:
    family_kind, _ = parse_seeded_experiment_name(experiment)
    if family_kind in {"nn", "resnet"}:
        return PLAIN_RESNET_NN_TRAINING_ARGS
    if family_kind in {"nnp", "resnetp"}:
        return COMMON_TRAINING_ARGS
    return ()


def supports_live_progress() -> bool:
    return sys.stdout.isatty() and os.environ.get("TERM", "dumb") != "dumb"


def make_default_cell(*, enabled: bool, blocking_batch: int, devices: list[str]) -> CellState:
    default_gpu = devices[0] if devices else ""
    return CellState(enabled=enabled, blocking_batch=blocking_batch, gpus=[default_gpu] if default_gpu else [])


def make_default_layer_state(exp_def: ExperimentDef, ensemble_ids: list[int], devices: list[str]) -> LayerState:
    return LayerState(
        cells={
            ensemble: make_default_cell(
                enabled=exp_def.default_enabled,
                blocking_batch=exp_def.default_blocking_batch,
                devices=devices,
            )
            for ensemble in ensemble_ids
        }
    )


def layer_is_enabled(layer_state: LayerState) -> bool:
    return any(cell.enabled for cell in layer_state.cells.values())


def experiment_is_enabled(row: ExperimentState) -> bool:
    return any(layer_is_enabled(layer_state) for layer_state in row.layer_states.values())


def normalize_enabled_blocking_batches(state: dict[str, Any]) -> None:
    next_batch = 1
    ensemble_ids = state["ensemble_ids"]
    for name in EXPERIMENT_ORDER:
        row = state["experiments"][name]
        layers = resolved_layers_for_state(name, row)
        blocking_batch_values = sorted(
            {
                row.layer_states[layer].cells[ensemble].blocking_batch
                for layer in layers
                for ensemble in ensemble_ids
                if row.layer_states[layer].cells[ensemble].enabled
            }
        )
        if not blocking_batch_values:
            continue
        mapping = {
            blocking_batch: next_batch + offset
            for offset, blocking_batch in enumerate(blocking_batch_values)
        }
        next_batch += len(blocking_batch_values)
        for layer in layers:
            for ensemble in ensemble_ids:
                cell = row.layer_states[layer].cells[ensemble]
                if cell.blocking_batch in mapping:
                    cell.blocking_batch = mapping[cell.blocking_batch]


def resolved_layers_for_state(experiment: str, row: ExperimentState) -> list[int]:
    layers = parse_integer_spec(row.layers_spec, "layer")
    validate_layers(experiment, layers)
    return layers


def sync_layer_cell_columns(
    layer_state: LayerState,
    ensemble_ids: list[int],
    devices: list[str],
    *,
    enabled_default: bool,
    blocking_batch_default: int,
) -> LayerState:
    updated_cells: dict[int, CellState] = {}
    for ensemble in ensemble_ids:
        if ensemble in layer_state.cells:
            updated_cells[ensemble] = layer_state.cells[ensemble]
        else:
            updated_cells[ensemble] = make_default_cell(
                enabled=enabled_default,
                blocking_batch=blocking_batch_default,
                devices=devices,
            )
    return LayerState(cells=updated_cells)


def sync_layer_states(experiment: str, row: ExperimentState, ensemble_ids: list[int], devices: list[str]) -> None:
    exp_def = EXPERIMENTS[experiment]
    layers = resolved_layers_for_state(experiment, row)
    updated_states: dict[int, LayerState] = {}
    for layer in layers:
        if layer in row.layer_states:
            layer_state = row.layer_states[layer]
            updated_states[layer] = sync_layer_cell_columns(
                layer_state,
                ensemble_ids,
                devices,
                enabled_default=layer_is_enabled(layer_state),
                blocking_batch_default=exp_def.default_blocking_batch,
            )
        else:
            updated_states[layer] = make_default_layer_state(exp_def, ensemble_ids, devices)
    row.layer_states = updated_states


def auto_plan_experiment(experiment: str, row: ExperimentState, ensemble_ids: list[int], devices: list[str]) -> None:
    if not devices:
        for layer in resolved_layers_for_state(experiment, row):
            layer_state = row.layer_states[layer]
            for ensemble in ensemble_ids:
                cell = layer_state.cells[ensemble]
                cell.gpus = []
                cell.blocking_batch = 1
        return

    batch = 1
    batch_counts: dict[str, int] = defaultdict(int)
    batch_caps: dict[str, int | None] = {}
    batch_loads: dict[str, float] = defaultdict(float)

    def start_new_batch() -> None:
        nonlocal batch, batch_counts, batch_caps, batch_loads
        batch += 1
        batch_counts = defaultdict(int)
        batch_caps = {}
        batch_loads = defaultdict(float)

    for layer in resolved_layers_for_state(experiment, row):
        layer_state = row.layer_states[layer]
        layer_cells = [layer_state.cells[ensemble] for ensemble in ensemble_ids]
        enabled_cells = [cell for cell in layer_cells if cell.enabled]

        if not enabled_cells:
            counts = assign_counts_by_load(
                devices,
                batch_counts={},
                batch_caps={},
                batch_loads={},
                layer=layer,
                total_cells=len(layer_cells),
                enforce_layer_cap=False,
            )
            if counts is None:
                raise RuntimeError(f"Could not assign GPUs for layer {layer}")
            assign_layer_chunk_to_devices(layer_cells, counts, batch)
            continue

        counts = assign_counts_by_load(
            devices,
            batch_counts=batch_counts,
            batch_caps=batch_caps,
            batch_loads=batch_loads,
            layer=layer,
            total_cells=len(enabled_cells),
        )
        if counts is None and batch_counts:
            start_new_batch()
            counts = assign_counts_by_load(
                devices,
                batch_counts=batch_counts,
                batch_caps=batch_caps,
                batch_loads=batch_loads,
                layer=layer,
                total_cells=len(enabled_cells),
            )

        if counts is not None:
            enabled_index = 0
            for device in devices:
                take = counts.get(device, 0)
                chunk = enabled_cells[enabled_index : enabled_index + take]
                for cell in chunk:
                    cell.gpus = [device]
                    cell.blocking_batch = batch
                enabled_index += take
                if take <= 0:
                    continue
                layer_cap = depth_concurrency_limit(layer, devices[0])
                batch_caps[device] = layer_cap if batch_caps.get(device) is None else min(int(batch_caps[device]), layer_cap)
                batch_counts[device] += take
                batch_loads[device] += take * layer_load_weight(layer)

            if enabled_index != len(enabled_cells):
                raise RuntimeError(f"Incomplete GPU assignment for layer {layer}")
        else:
            layer_cap = depth_concurrency_limit(layer, devices[0])
            remaining_cells = enabled_cells[:]
            while remaining_cells:
                if batch_counts:
                    start_new_batch()
                chunk_counts = assign_counts_by_load(
                    devices,
                    batch_counts={},
                    batch_caps={},
                    batch_loads={},
                    layer=layer,
                    total_cells=min(len(remaining_cells), len(devices) * layer_cap),
                )
                if chunk_counts is None:
                    raise RuntimeError(f"Could not assign GPU chunk for layer {layer}")
                planned = sum(chunk_counts.values())
                chunk = remaining_cells[:planned]
                enabled_index = 0
                for device in devices:
                    take = chunk_counts.get(device, 0)
                    device_chunk = chunk[enabled_index : enabled_index + take]
                    for cell in device_chunk:
                        cell.gpus = [device]
                        cell.blocking_batch = batch
                    enabled_index += take
                    if take <= 0:
                        continue
                    batch_caps[device] = layer_cap
                    batch_counts[device] += take
                    batch_loads[device] += take * layer_load_weight(layer)
                remaining_cells = remaining_cells[planned:]

        disabled_cells = [cell for cell in layer_cells if not cell.enabled]
        if disabled_cells:
            counts = assign_counts_by_load(
                devices,
                batch_counts={},
                batch_caps={},
                batch_loads={},
                layer=layer,
                total_cells=len(disabled_cells),
                enforce_layer_cap=False,
            )
            if counts is None:
                raise RuntimeError(f"Could not assign GPUs for disabled cells in layer {layer}")
            assign_layer_chunk_to_devices(disabled_cells, counts, batch)


def default_state(devices: list[str]) -> dict[str, Any]:
    ensemble_ids = parse_integer_spec(DEFAULT_ENSEMBLE_SPEC, "ensemble", allow_desc=False)
    experiments: dict[str, ExperimentState] = {}
    for name in EXPERIMENT_ORDER:
        exp_def = EXPERIMENTS[name]
        row = ExperimentState(layers_spec=exp_def.default_layers, layer_states={})
        sync_layer_states(name, row, ensemble_ids, devices)
        auto_plan_experiment(name, row, ensemble_ids, devices)
        experiments[name] = row
    state = {
        "devices": list(devices),
        "gpu_slots": {device: default_slots(device) for device in devices},
        "ensemble_ids": ensemble_ids,
        "experiments": experiments,
    }
    normalize_enabled_blocking_batches(state)
    return state


def sync_ensemble_columns(state: dict[str, Any], ensemble_ids: list[int]) -> None:
    state["ensemble_ids"] = list(ensemble_ids)
    for name in EXPERIMENT_ORDER:
        row = state["experiments"][name]
        sync_layer_states(name, row, ensemble_ids, state["devices"])
        auto_plan_experiment(name, row, ensemble_ids, state["devices"])
    normalize_enabled_blocking_batches(state)


def build_plan_payload(state: dict[str, Any]) -> dict[str, Any]:
    selected_any = False
    experiments_payload: list[dict[str, Any]] = []
    for name in EXPERIMENT_ORDER:
        row = state["experiments"][name]
        layers = resolved_layers_for_state(name, row)
        layer_payloads: list[dict[str, Any]] = []
        experiment_enabled = False
        for layer in layers:
            layer_state = row.layer_states[layer]
            cells_payload: list[dict[str, Any]] = []
            layer_enabled = False
            selected_ensembles: list[int] = []
            for ensemble in state["ensemble_ids"]:
                cell = layer_state.cells[ensemble]
                if cell.enabled:
                    if not cell.gpus:
                        raise ValueError(f"{name} layer {layer} ensemble {ensemble} has no GPU preference")
                    if len(cell.gpus) != 1:
                        raise ValueError(f"{name} layer {layer} ensemble {ensemble} must have exactly one preferred device")
                    selected_any = True
                    layer_enabled = True
                    experiment_enabled = True
                    selected_ensembles.append(int(ensemble))
                cells_payload.append(
                    {
                        "ensemble": ensemble,
                        "enabled": bool(cell.enabled),
                        "blocking_batch": int(cell.blocking_batch),
                        "gpus": list(cell.gpus),
                    }
                )
            if selected_ensembles:
                selected_ensemble_range(selected_ensembles, experiment=name, layer=layer)
            layer_payloads.append(
                {
                    "layer": int(layer),
                    "enabled": bool(layer_enabled),
                    "cells": cells_payload,
                }
            )

        experiments_payload.append(
            {
                "name": name,
                "enabled": bool(experiment_enabled),
                "layers_spec": row.layers_spec,
                "layers": layer_payloads,
            }
        )

    if not selected_any:
        raise ValueError("No experiment-ensemble cells are selected")
    if not state["devices"]:
        raise ValueError("No compute devices are available")

    return {
        "version": 1,
        "created_at": int(time.time()),
        "devices": list(state["devices"]),
        "gpu_slots": normalized_gpu_slots(state.get("gpu_slots"), state["devices"]),
        "ensemble_ids": list(state["ensemble_ids"]),
        "run_econ": True,
        "group_ablation_only": False,
        "use_pretrained": bool(state.get("use_pretrained", False)),
        "experiments": experiments_payload,
    }


def payload_layer_entries(experiment_payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_layers = experiment_payload.get("layers", [])
    if raw_layers and isinstance(raw_layers[0], int):
        return [
            {
                "layer": int(layer),
                "enabled": bool(experiment_payload.get("enabled", True)),
                "cells": experiment_payload.get("cells", []),
            }
            for layer in raw_layers
        ]
    return [entry for entry in raw_layers if isinstance(entry, dict)]


def payload_runs_econ(payload: dict[str, Any]) -> bool:
    return payload.get("run_econ") is True


def payload_runs_group_ablation_only(payload: dict[str, Any]) -> bool:
    return payload.get("group_ablation_only") is True


def resolve_payload_gpu_list(payload_gpus: list[Any], current_devices: list[str], payload_devices: list[str]) -> list[str]:
    if current_devices in (["cpu"], ["mps"]):
        return list(current_devices)
    resolved: list[str] = []
    seen: set[str] = set()
    for gpu in payload_gpus:
        token = str(gpu)
        mapped: str | None = None
        if token in current_devices:
            mapped = token
        elif token in payload_devices:
            position = payload_devices.index(token)
            if 0 <= position < len(current_devices):
                mapped = current_devices[position]
        elif token.isdigit():
            position = int(token)
            if 0 <= position < len(current_devices):
                mapped = current_devices[position]
        if mapped is not None and mapped not in seen:
            seen.add(mapped)
            resolved.append(mapped)
    return resolved


def resolve_payload_gpu_slots(
    payload_gpu_slots: dict[str, Any] | None,
    current_devices: list[str],
    payload_devices: list[str],
) -> dict[str, int]:
    raw = payload_gpu_slots or {}
    resolved: dict[str, int] = {}
    for index, device in enumerate(current_devices):
        candidate = raw.get(device)
        if candidate is None and index < len(payload_devices):
            candidate = raw.get(payload_devices[index])
        if candidate is None:
            resolved[device] = default_slots(device)
            continue
        resolved[device] = device_slots(device, candidate)
    return resolved


def state_from_payload(payload: dict[str, Any], current_devices: list[str]) -> dict[str, Any]:
    if not current_devices:
        raise ValueError("No compute devices are available")

    payload_devices = [str(device) for device in payload.get("devices", current_devices)]
    raw_ensemble_ids = payload.get("ensemble_ids", parse_integer_spec(DEFAULT_ENSEMBLE_SPEC, "ensemble", allow_desc=False))
    ensemble_ids = [int(ensemble) for ensemble in raw_ensemble_ids]
    if not ensemble_ids:
        raise ValueError("Plan ensemble_ids cannot be empty")

    state = default_state(current_devices)
    state["use_pretrained"] = bool(payload.get("use_pretrained", False))
    sync_ensemble_columns(state, ensemble_ids)
    state["gpu_slots"] = resolve_payload_gpu_slots(payload.get("gpu_slots"), current_devices, payload_devices)

    payload_by_name = {
        str(experiment.get("name")): experiment
        for experiment in payload.get("experiments", [])
        if isinstance(experiment, dict) and str(experiment.get("name")) in EXPERIMENT_ORDER
    }

    for name in EXPERIMENT_ORDER:
        row = state["experiments"][name]
        experiment_payload = payload_by_name.get(name)
        if experiment_payload is None:
            for layer_state in row.layer_states.values():
                for cell in layer_state.cells.values():
                    cell.enabled = False
            continue

        layer_entries = payload_layer_entries(experiment_payload)
        payload_layers = [int(entry["layer"]) for entry in layer_entries if "layer" in entry]
        if "layers_spec" in experiment_payload:
            row.layers_spec = str(experiment_payload["layers_spec"])
        elif payload_layers:
            row.layers_spec = compact_integer_spec(sorted(payload_layers))
        sync_layer_states(name, row, ensemble_ids, current_devices)

        for layer_state in row.layer_states.values():
            for ensemble in ensemble_ids:
                layer_state.cells[ensemble].enabled = False

        for layer_entry in layer_entries:
            if "layer" not in layer_entry:
                continue
            layer = int(layer_entry["layer"])
            if layer not in row.layer_states:
                continue
            cells_by_ensemble = {
                int(cell_payload["ensemble"]): cell_payload
                for cell_payload in layer_entry.get("cells", [])
                if isinstance(cell_payload, dict) and "ensemble" in cell_payload
            }
            for ensemble, payload_cell in cells_by_ensemble.items():
                if ensemble not in row.layer_states[layer].cells:
                    continue
                cell = row.layer_states[layer].cells[ensemble]
                cell.enabled = bool(payload_cell.get("enabled", False))
                cell.blocking_batch = payload_blocking_batch(payload_cell)
                resolved_gpus = resolve_payload_gpu_list(
                    list(payload_cell.get("gpus", [])),
                    current_devices,
                    payload_devices,
                )
                if cell.enabled and not resolved_gpus:
                    raise ValueError(f"{name} layer {layer} ensemble {ensemble} has no usable device in this environment")
                if resolved_gpus:
                    cell.gpus = resolved_gpus

    normalize_enabled_blocking_batches(state)
    return state


def payload_blocking_batch(cell: dict[str, Any]) -> int:
    if "blocking_batch" in cell:
        return int(cell["blocking_batch"])
    if "stage" in cell:
        return int(cell["stage"])
    raise KeyError("Plan cell is missing blocking_batch")


def validate_runtime_inputs(project_root: Path, *, require_cluster_labels: bool = False) -> None:
    required = [
        project_root / "main.py",
        project_root / "utils" / "prepare_data.py",
        project_root / "source_data" / "datashare_with_return.pkl",
        project_root / "source_data" / "PredictorData2024.xlsx",
        project_root / "source_data" / "F-F_Research_Data_5_Factors_2x3.csv",
    ]
    if require_cluster_labels:
        required.append(project_root / "source_data" / "cluster_labels.csv")
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")


def validate_launcher_paths(project_root: Path) -> None:
    for path in (project_root / "main.py", project_root / "utils" / "prepare_data.py"):
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")


def has_clean_cache(project_root: Path) -> bool:
    return all(
        (project_root / "tmp" / name).exists()
        for name in ("clean_X.pkl", "clean_y.pkl", "clean_mvel1.pkl", "clean_ff5.pkl")
    )


def has_presplit_cache(project_root: Path) -> bool:
    tmp_root = project_root / "tmp"
    if not tmp_root.exists():
        return False
    return any(tmp_root.glob("*/X_train_tensor.pt"))


def run_subprocess(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    dry_run: bool = False,
) -> None:
    if dry_run:
        prefix = ""
        if env and "CUDA_VISIBLE_DEVICES" in env:
            prefix = f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} "
        print(f"[dry-run] {prefix}{' '.join(shlex.quote(part) for part in command)}")
        return

    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    subprocess.run(command, check=True, cwd=str(cwd), env=merged_env)


def ensure_runtime_ready(
    *,
    project_root: Path,
    python_bin: str,
    main_py: Path,
    prepare_data_py: Path,
    dry_run: bool,
    group_ablation_only: bool = False,
) -> bool:
    validate_launcher_paths(project_root)
    if not dry_run:
        validate_runtime_inputs(project_root, require_cluster_labels=group_ablation_only)

    if not has_clean_cache(project_root):
        run_subprocess(
            [python_bin, str(prepare_data_py), "--project-root", str(project_root)],
            cwd=project_root,
            dry_run=dry_run,
        )

    use_presplit = has_presplit_cache(project_root)
    if not use_presplit:
        run_subprocess(
            [
                python_bin,
                str(main_py),
                "--config_cls",
                "ResNetpS4D5Config",
                "--asset_prefix",
                "pre_split_init",
                "--init_pre_split_dataset",
                "--device",
                "cpu",
            ],
            cwd=project_root,
            dry_run=dry_run,
        )
        use_presplit = True

    return use_presplit


def group_ablation_layers_with_anchor(
    experiment: str, layer_entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Include one shallow anchor with the same ensemble as the selected depths."""
    _, anchor_depth = parse_seeded_experiment_name(experiment)
    active = [entry for entry in layer_entries if entry["enabled"]
              and any(cell["enabled"] for cell in entry["cells"])]
    if not active:
        return []
    ensemble_ranges = {
        selected_ensemble_range(
            [int(cell["ensemble"]) for cell in entry["cells"] if cell["enabled"]],
            experiment=experiment, layer=int(entry["layer"]),
        )
        for entry in active
    }
    if len(ensemble_ranges) != 1:
        raise ValueError(
            f"{experiment}: group ablation requires the same ensemble range across "
            "selected depths so they can share one shallow-anchor output."
        )
    anchor = next((entry for entry in active if int(entry["layer"]) == anchor_depth), None)
    if anchor is None:
        anchor = {
            "layer": anchor_depth,
            "enabled": True,
            "cells": [dict(cell) for cell in active[0]["cells"]],
        }
    return [anchor, *[entry for entry in active if int(entry["layer"]) != anchor_depth]]


def build_execution_jobs(
    payload: dict[str, Any],
    *,
    project_root: Path,
    python_bin: str,
    main_py: Path,
    use_presplit: bool,
) -> tuple[dict[int, list[ScheduledGpuJob]], dict[int, list[ScheduledForegroundJob]]]:
    jobs_by_blocking_batch: dict[int, list[ScheduledGpuJob]] = defaultdict(list)
    foreground_by_blocking_batch: dict[int, list[ScheduledForegroundJob]] = defaultdict(list)
    group_ablation_only = payload_runs_group_ablation_only(payload)
    run_econ = payload_runs_econ(payload) and not group_ablation_only

    for exp_index, experiment in enumerate(payload["experiments"]):
        if not experiment["enabled"]:
            continue

        exp_name = experiment["name"]
        exp_def = EXPERIMENTS[exp_name]
        experiment_kind, _ = parse_seeded_experiment_name(exp_name)
        if group_ablation_only and experiment_kind not in GROUP_ABLATION_KINDS:
            continue
        raw_layers = experiment["layers"]
        if raw_layers and isinstance(raw_layers[0], int):
            layer_entries = [
                {
                    "layer": int(layer),
                    "enabled": bool(experiment.get("enabled", True)),
                    "cells": experiment["cells"],
                }
                for layer in raw_layers
            ]
        else:
            layer_entries = raw_layers

        if group_ablation_only:
            layer_entries = group_ablation_layers_with_anchor(exp_name, layer_entries)

        for layer_index, layer_entry in enumerate(layer_entries):
            if not layer_entry["enabled"]:
                continue

            selected_cells = [cell for cell in layer_entry["cells"] if cell["enabled"]]
            if not selected_cells:
                continue

            layer = int(layer_entry["layer"])
            config_cls = config_for_experiment(exp_name, layer)
            asset_prefix = asset_prefix_for_experiment(exp_name, layer)
            if payload.get("use_pretrained"):
                asset_prefix = f"pretrained_{asset_prefix}"
            max_blocking_batch = max(payload_blocking_batch(cell) for cell in selected_cells)
            selected_ensembles = [int(cell["ensemble"]) for cell in selected_cells]
            ensemble_start, ensemble_end = selected_ensemble_range(
                selected_ensembles,
                experiment=exp_name,
                layer=layer,
            )

            extra_args = list(extra_args_for_experiment(exp_name))
            completion_key = (exp_index, layer_index)
            if group_ablation_only:
                assigned_gpu = str(selected_cells[layer_index % len(selected_cells)]["gpus"][0])
                command = [
                    python_bin,
                    str(main_py),
                    "--config_cls",
                    config_cls,
                    "--asset_prefix",
                    asset_prefix,
                    "--group_ablation",
                    "--device",
                    worker_device(assigned_gpu),
                    "--use_pre_split_dataset",
                    "--start_ensemble",
                    str(ensemble_start),
                    "--max_ensemble",
                    str(ensemble_end),
                    "--econ_mkt_cap",
                    "ALL",
                    *ECON_WEIGHT_ARGS,
                    *extra_args,
                ]
                jobs_by_blocking_batch[1].append(
                    ScheduledGpuJob(
                        experiment_index=exp_index,
                        blocking_batch=1,
                        layer=layer,
                        assigned_gpu=assigned_gpu,
                        command=command,
                        label=f"{asset_prefix} group-ablation",
                        progress_group=asset_prefix,
                        progress_item="group-ablation",
                        completion_key=completion_key,
                        order_key=(layer_index, exp_index, 0),
                    )
                )
                continue

            for cell_index, cell in enumerate(selected_cells):
                ensemble = int(cell["ensemble"])
                assigned_gpu = str(cell["gpus"][0])
                command = [
                    python_bin,
                    str(main_py),
                    "--config_cls",
                    config_cls,
                    "--asset_prefix",
                    asset_prefix,
                    *([] if payload.get("use_pretrained") else ["--train"]),
                    "--predict",
                    "--ensemble",
                    str(ensemble),
                    "--device",
                    worker_device(assigned_gpu),
                ]
                if use_presplit:
                    command.append("--use_pre_split_dataset")
                command.extend(extra_args)
                jobs_by_blocking_batch[payload_blocking_batch(cell)].append(
                    ScheduledGpuJob(
                        experiment_index=exp_index,
                        blocking_batch=payload_blocking_batch(cell),
                        layer=layer,
                        assigned_gpu=assigned_gpu,
                        command=command,
                        label=f"{asset_prefix} e{ensemble:02d}",
                        progress_group=asset_prefix,
                        progress_item=f"e{ensemble:02d}",
                        completion_key=completion_key,
                        order_key=(exp_index, layer_index, cell_index),
                    )
                )

            if run_econ:
                foreground_by_blocking_batch[max_blocking_batch].append(
                    ScheduledForegroundJob(
                        experiment_index=exp_index,
                        blocking_batch=max_blocking_batch,
                        command=[
                            python_bin,
                            str(main_py),
                            "--config_cls",
                            config_cls,
                            "--asset_prefix",
                            asset_prefix,
                            "--econ",
                            "--device",
                            "cpu",
                            "--start_ensemble",
                            str(ensemble_start),
                            "--max_ensemble",
                            str(ensemble_end),
                            "--econ_mkt_cap",
                            *REPORT_GROUPS,
                            *ECON_WEIGHT_ARGS,
                            *extra_args,
                        ],
                        label=f"{asset_prefix} econ",
                        depends_on=completion_key,
                        order_key=(exp_index, layer_index, 0, 0),
                    )
                )

    if group_ablation_only and not jobs_by_blocking_batch:
        raise ValueError("Group ablation requires at least one enabled ResNet, NN, ResNet+, or NN+ model-depth cell.")

    for stage_jobs in jobs_by_blocking_batch.values():
        stage_jobs.sort(key=lambda job: job.order_key)
    for foreground_jobs in foreground_by_blocking_batch.values():
        foreground_jobs.sort(key=lambda job: job.order_key)

    return jobs_by_blocking_batch, foreground_by_blocking_batch


def progress_title(payload: dict[str, Any]) -> str:
    ensemble_text = compact_integer_spec(payload["ensemble_ids"])
    mode = "group-ablation-only" if payload_runs_group_ablation_only(payload) else "train+predict"
    if payload.get("use_pretrained") and not payload_runs_group_ablation_only(payload):
        mode = "pretrained predict"
    return f"ResAssetPricing launcher | {mode} | ens {ensemble_text}"


def refresh_report_sources(*, project_root, python_bin, ensemble_ids, dry_run):
    """Collect numerical portfolio outputs into a separate replication input folder."""
    run_subprocess([python_bin, "-m", "utils.paper_source.collect",
                    "--output", str(project_root / "tmp" / f"replication_inputs_{time.strftime('%Y%m%d_%H%M%S')}")],
                   cwd=project_root, dry_run=dry_run)


def blocking_batch_progress_title(payload: dict[str, Any], blocking_batch: int, jobs: list[ScheduledGpuJob]) -> str:
    base = progress_title(payload)
    groups = len({job.progress_group for job in jobs})
    return f"{base} | blocking-batch {blocking_batch:02d} | groups={groups} | jobs={len(jobs)}"


def execute_stage_jobs(
    jobs: list[ScheduledGpuJob],
    *,
    project_root: Path,
    devices: list[str],
    gpu_slots: dict[str, int],
    session_id: str | None,
    session_title: str | None,
    dry_run: bool,
    on_tick: Callable[[], None] | None = None,
) -> None:
    if not jobs:
        return

    gpu_limits = normalized_gpu_slots(gpu_slots, devices)
    pending = list(jobs)
    job_slots = {id(job): index for index, job in enumerate(jobs)}
    active: dict[int, tuple[subprocess.Popen[str], ScheduledGpuJob]] = {}
    active_counts: dict[str, int] = defaultdict(int)
    active_layer_limits: dict[str, list[int]] = defaultdict(list)

    def emit(message: str) -> None:
        if session_id and supports_live_progress():
            append_event(session_id, message)
        else:
            print(message)

    def launch(job: ScheduledGpuJob) -> subprocess.Popen[str]:
        gpu = job.assigned_gpu
        env = os.environ.copy()
        env.update(worker_environment(gpu))
        if session_id and supports_live_progress():
            env["ResAssetPricing_PROGRESS_SESSION"] = session_id
            env["ResAssetPricing_PROGRESS_SLOT"] = str(job_slots[id(job)])
            env["ResAssetPricing_PROGRESS_GROUP_LABEL"] = job.progress_group
            env["ResAssetPricing_PROGRESS_ROW_LABEL"] = f"  {job.progress_item}"
            env["ResAssetPricing_PROGRESS_LABEL"] = f"device {gpu}"
        emit(f"[launch] blocking-batch={job.blocking_batch} gpu={gpu} {job.label}")
        return subprocess.Popen(job.command, cwd=str(project_root), env=env, text=True)

    if dry_run:
        for job in pending:
            env = worker_environment(job.assigned_gpu)
            run_subprocess(job.command, cwd=project_root, env=env, dry_run=True)
        return

    if session_id and supports_live_progress():
        init_session(
            session_id,
            len(jobs),
            session_title or f"blocking-batch {jobs[0].blocking_batch:02d}",
            row_templates=[
                {
                    "group_label": job.progress_group,
                    "row_label": f"  {job.progress_item}",
                    "label": f"device {job.assigned_gpu}",
                    "detail": "waiting for worker",
                }
                for job in jobs
            ],
        )

    try:
        while pending or active:
            if on_tick is not None:
                on_tick()
            launched = False
            index = 0
            while index < len(pending):
                job = pending[index]
                if job.assigned_gpu not in devices:
                    index += 1
                    continue
                gpu = job.assigned_gpu
                if active_counts[gpu] >= effective_gpu_total_limit(
                    gpu,
                    candidate_layer=job.layer,
                    gpu_limits=gpu_limits,
                    active_layer_limits=active_layer_limits,
                ):
                    index += 1
                    continue
                process = launch(job)
                active[id(process)] = (process, job)
                active_counts[gpu] += 1
                active_layer_limits[gpu].append(depth_concurrency_limit(job.layer))
                pending.pop(index)
                launched = True

            if not active and pending and not launched:
                raise RuntimeError("No schedulable device jobs remain; check blocking-batch device assignments and per-device total caps")

            time.sleep(0.2)
            if on_tick is not None:
                on_tick()
            for process_key, (process, job) in list(active.items()):
                status = process.poll()
                if status is None:
                    continue
                del active[process_key]
                active_counts[job.assigned_gpu] -= 1
                try:
                    active_layer_limits[job.assigned_gpu].remove(depth_concurrency_limit(job.layer))
                except ValueError:
                    pass
                if status != 0:
                    for other_process, _ in active.values():
                        other_process.terminate()
                    raise subprocess.CalledProcessError(status, job.command)
                emit(f"[done] blocking-batch={job.blocking_batch} gpu={job.assigned_gpu} {job.label}")
    finally:
        for process, _ in active.values():
            process.terminate()


def execute_gpu_jobs_by_blocking_batch(
    jobs_by_blocking_batch: dict[int, list[ScheduledGpuJob]],
    *,
    payload: dict[str, Any],
    foreground_by_blocking_batch: dict[int, list[ScheduledForegroundJob]],
    project_root: Path,
    devices: list[str],
    gpu_slots: dict[str, int],
    session_id: str | None,
    dry_run: bool,
) -> None:
    if not jobs_by_blocking_batch and not foreground_by_blocking_batch:
        return

    gpu_jobs = sorted(
        (job for jobs in jobs_by_blocking_batch.values() for job in jobs),
        key=lambda job: (job.blocking_batch, *job.order_key),
    )
    foreground_jobs = sorted(
        (job for jobs in foreground_by_blocking_batch.values() for job in jobs),
        key=lambda job: (job.blocking_batch, *job.order_key),
    )
    batch_group_labels: dict[int, list[str]] = defaultdict(list)
    for job in gpu_jobs:
        if job.progress_group not in batch_group_labels[job.blocking_batch]:
            batch_group_labels[job.blocking_batch].append(job.progress_group)

    def emit(message: str) -> None:
        if session_id and supports_live_progress():
            append_event(session_id, message)
        else:
            print(message)

    if dry_run:
        for blocking_batch in sorted(set(jobs_by_blocking_batch) | set(foreground_by_blocking_batch)):
            jobs = sorted(jobs_by_blocking_batch.get(blocking_batch, []), key=lambda job: job.order_key)
            foreground_jobs_for_batch = sorted(
                foreground_by_blocking_batch.get(blocking_batch, []),
                key=lambda job: job.order_key,
            )
            group_names = ",".join(dict.fromkeys(job.progress_group for job in jobs))
            if jobs:
                emit(
                    f"[blocking-batch-open] blocking-batch={blocking_batch:02d} "
                    f"groups={group_names} jobs={len(jobs)}"
                )
            for job in jobs:
                run_subprocess(job.command, cwd=project_root, env=worker_environment(job.assigned_gpu), dry_run=True)
            for job in foreground_jobs_for_batch:
                run_subprocess(job.command, cwd=project_root, dry_run=True)
            if jobs:
                emit(f"[blocking-batch-done] blocking-batch={blocking_batch:02d} groups={group_names}")
        return

    foreground_manager = AsyncForegroundJobManager(
        project_root=project_root,
        session_id=session_id,
        dry_run=dry_run,
    )
    gpu_limits = normalized_gpu_slots(gpu_slots, devices)
    group_ablation_only = payload_runs_group_ablation_only(payload)
    pending = list(gpu_jobs)
    pending_foreground = list(foreground_jobs)
    remaining_by_completion = Counter(job.completion_key for job in pending)
    remaining_by_batch = Counter(job.blocking_batch for job in pending)
    active: dict[int, tuple[subprocess.Popen[str], ScheduledGpuJob, str]] = {}
    active_counts: dict[str, int] = defaultdict(int)
    gpu_done_fds: dict[int, int] = {}
    released_gpu_slots: set[int] = set()
    active_layer_limits: dict[str, list[int]] = defaultdict(list)
    announced_batches: set[int] = set()
    completed_batches: set[int] = set()
    opened_completion_keys: set[tuple[int, ...]] = set()

    row_slots: dict[tuple[str, str], int] = {}
    rows_by_completion: dict[tuple[int, ...], list[dict[str, str | int]]] = defaultdict(list)
    for job in gpu_jobs:
        row_key = (job.progress_group, job.progress_item)
        if row_key not in row_slots:
            row_slots[row_key] = len(row_slots)
        slot = row_slots[row_key]
        rows_by_completion[job.completion_key].append(
            {
                "slot": slot,
                "group_label": job.progress_group,
                "row_label": f"  {job.progress_item}",
                "label": f"device {job.assigned_gpu}",
                "detail": "waiting for worker",
            }
        )

    def ensure_completion_rows(job: ScheduledGpuJob) -> None:
        if not session_id or not supports_live_progress() or job.completion_key in opened_completion_keys:
            return
        ensure_row_templates(session_id, rows_by_completion[job.completion_key])
        opened_completion_keys.add(job.completion_key)

    def announce_batch(job: ScheduledGpuJob) -> None:
        if job.blocking_batch in announced_batches:
            return
        groups = ",".join(batch_group_labels.get(job.blocking_batch, []))
        emit(
            f"[blocking-batch-open] blocking-batch={job.blocking_batch:02d} "
            f"groups={groups} jobs={remaining_by_batch[job.blocking_batch]}"
        )
        announced_batches.add(job.blocking_batch)

    def emit_completed_batches() -> None:
        for blocking_batch in sorted(announced_batches):
            if blocking_batch in completed_batches or remaining_by_batch[blocking_batch] > 0:
                continue
            groups = ",".join(batch_group_labels.get(blocking_batch, []))
            emit(f"[blocking-batch-done] blocking-batch={blocking_batch:02d} groups={groups}")
            completed_batches.add(blocking_batch)

    def spare_capacity(gpu: str, job: ScheduledGpuJob) -> int:
        if group_ablation_only:
            limit = gpu_limits.get(gpu, GROUP_ABLATION_GPU_CONCURRENCY)
            # Bound CPU econ/save backlog as well as concurrent GPU inference.
            if sum(active_gpu == gpu for _, _, active_gpu in active.values()) >= 2 * limit:
                return 0
        else:
            limit = effective_gpu_total_limit(
                gpu,
                candidate_layer=job.layer,
                gpu_limits=gpu_limits,
                active_layer_limits=active_layer_limits,
            )
        return limit - active_counts[gpu]

    def choose_gpu(job: ScheduledGpuJob) -> str | None:
        best: tuple[tuple[int, int, int, int], str] | None = None
        for device_index, gpu in enumerate(devices):
            spare = spare_capacity(gpu, job)
            if spare <= 0:
                continue
            preference = 0 if gpu == job.assigned_gpu else 1
            score = (-spare, active_counts[gpu], preference, device_index)
            if best is None or score < best[0]:
                best = (score, gpu)
        return best[1] if best is not None else None

    def launch(job: ScheduledGpuJob, gpu: str) -> subprocess.Popen[str]:
        env = os.environ.copy()
        env.update(worker_environment(gpu))
        slot = row_slots[(job.progress_group, job.progress_item)]
        if session_id and supports_live_progress():
            env["ResAssetPricing_PROGRESS_SESSION"] = session_id
            env["ResAssetPricing_PROGRESS_SLOT"] = str(slot)
            env["ResAssetPricing_PROGRESS_GROUP_LABEL"] = job.progress_group
            env["ResAssetPricing_PROGRESS_ROW_LABEL"] = f"  {job.progress_item}"
            env["ResAssetPricing_PROGRESS_LABEL"] = f"device {gpu}"
        if gpu == job.assigned_gpu:
            emit(f"[launch] blocking-batch={job.blocking_batch} gpu={gpu} {job.label}")
        else:
            emit(
                f"[launch] blocking-batch={job.blocking_batch} "
                f"gpu={gpu} preferred-gpu={job.assigned_gpu} {job.label}"
            )
        if session_id and supports_live_progress():
            update_row_state(
                session_id,
                {
                    "slot": slot,
                    "status": "running",
                    "group_label": job.progress_group,
                    "row_label": f"  {job.progress_item}",
                    "label": f"device {gpu}",
                    "phase": "launch",
                    "current": 0.0,
                    "total": 1.0,
                    "detail": "worker started",
                    "metric_text": "",
                },
            )
        if group_ablation_only and gpu not in {"cpu", "mps"} and os.name != "nt":
            read_fd, write_fd = os.pipe()
            try:
                os.set_blocking(read_fd, False)
                env["APRNetRewrite_GPU_DONE_FD"] = str(write_fd)
                process = subprocess.Popen(
                    job.command, cwd=str(project_root), env=env, text=True,
                    pass_fds=(write_fd,),
                )
            except BaseException:
                os.close(read_fd)
                raise
            finally:
                os.close(write_fd)
            gpu_done_fds[id(process)] = read_fd
        else:
            process = subprocess.Popen(job.command, cwd=str(project_root), env=env, text=True)
        return process

    def launch_ready_foreground() -> None:
        ready: list[ScheduledForegroundJob] = []
        still_pending: list[ScheduledForegroundJob] = []
        for job in pending_foreground:
            if remaining_by_completion[job.depends_on] <= 0:
                ready.append(job)
            else:
                still_pending.append(job)
        if ready:
            if devices in (["cpu"], ["mps"]):
                for job in ready:
                    run_subprocess(job.command, cwd=project_root)
            else:
                foreground_manager.launch_jobs(ready)
        pending_foreground[:] = still_pending

    try:
        while pending or active or pending_foreground:
            foreground_manager.poll()
            launch_ready_foreground()
            launched = False

            index = 0
            while index < len(pending):
                job = pending[index]
                gpu = choose_gpu(job)
                if gpu is None:
                    index += 1
                    continue
                ensure_completion_rows(job)
                announce_batch(job)
                process = launch(job, gpu)
                active[id(process)] = (process, job, gpu)
                active_counts[gpu] += 1
                if not group_ablation_only:
                    active_layer_limits[gpu].append(depth_concurrency_limit(job.layer))
                pending.pop(index)
                launched = True

            if not active and pending and not launched:
                raise RuntimeError("No schedulable device jobs remain; check per-device caps and device availability")

            if not active and not pending:
                if pending_foreground and not any(remaining_by_completion[job.depends_on] <= 0 for job in pending_foreground):
                    raise RuntimeError("No device jobs remain, but some foreground jobs still have unmet dependencies")
                continue

            time.sleep(0.2)
            for process_key, (process, job, gpu) in list(active.items()):
                status = process.poll()
                if status is None:
                    read_fd = gpu_done_fds.get(process_key)
                    if read_fd is not None:
                        try:
                            gpu_done = os.read(read_fd, 1)
                        except BlockingIOError:
                            continue
                        os.close(gpu_done_fds.pop(process_key))
                        if gpu_done:
                            active_counts[gpu] -= 1
                            released_gpu_slots.add(process_key)
                            emit(f"[gpu-released] gpu={gpu} {job.label}; CPU econ/save continues")
                    continue
                del active[process_key]
                read_fd = gpu_done_fds.pop(process_key, None)
                if read_fd is not None:
                    os.close(read_fd)
                if process_key not in released_gpu_slots:
                    active_counts[gpu] -= 1
                released_gpu_slots.discard(process_key)
                if not group_ablation_only:
                    try:
                        active_layer_limits[gpu].remove(depth_concurrency_limit(job.layer))
                    except ValueError:
                        pass
                if status != 0:
                    for other_process, _, _ in active.values():
                        other_process.terminate()
                    raise subprocess.CalledProcessError(status, job.command)
                remaining_by_completion[job.completion_key] -= 1
                remaining_by_batch[job.blocking_batch] -= 1
                emit(f"[done] blocking-batch={job.blocking_batch} gpu={gpu} {job.label}")
            emit_completed_batches()

        foreground_manager.wait()
    except BaseException:
        for process, _, _ in active.values():
            process.terminate()
        foreground_manager.terminate_all()
        foreground_manager.reap_all()
        raise
    finally:
        for read_fd in gpu_done_fds.values():
            os.close(read_fd)


def execute_gpu_jobs_interleaved(
    jobs: list[ScheduledGpuJob],
    *,
    project_root: Path,
    devices: list[str],
    gpu_slots: dict[str, int],
    session_id: str | None,
    dry_run: bool,
) -> None:
    if not jobs:
        return

    ordered_jobs = sorted(jobs, key=lambda job: job.order_key)
    if dry_run:
        for job in ordered_jobs:
            env = worker_environment(job.assigned_gpu)
            run_subprocess(job.command, cwd=project_root, env=env, dry_run=True)
        return

    gpu_limits = normalized_gpu_slots(gpu_slots, devices)
    pending_by_batch: dict[tuple[int, int], list[ScheduledGpuJob]] = defaultdict(list)
    batch_order: dict[int, list[int]] = defaultdict(list)
    batch_gpu_limits: dict[tuple[int, int, str], int] = {}
    batch_group_labels: dict[tuple[int, int], list[str]] = defaultdict(list)
    job_slots = {id(job): index for index, job in enumerate(ordered_jobs)}

    for job in ordered_jobs:
        batch_key = (job.experiment_index, job.blocking_batch)
        pending_by_batch[batch_key].append(job)
        if job.blocking_batch not in batch_order[job.experiment_index]:
            batch_order[job.experiment_index].append(job.blocking_batch)
        if job.progress_group not in batch_group_labels[batch_key]:
            batch_group_labels[batch_key].append(job.progress_group)

    for batch_key, batch_jobs in pending_by_batch.items():
        for gpu in {job.assigned_gpu for job in batch_jobs}:
            batch_gpu_limits[(batch_key[0], batch_key[1], gpu)] = max(
                1,
                min(
                    gpu_limits.get(gpu, 20),
                    min(depth_concurrency_limit(job.layer, gpu) for job in batch_jobs if job.assigned_gpu == gpu),
                ),
            )

    current_batch_index = {exp_index: 0 for exp_index in batch_order}
    active: dict[int, tuple[subprocess.Popen[str], ScheduledGpuJob]] = {}
    active_counts: dict[str, int] = defaultdict(int)
    active_layer_limits: dict[str, list[int]] = defaultdict(list)
    active_batch_counts: dict[tuple[int, int, str], int] = defaultdict(int)
    active_batch_jobs: dict[tuple[int, int], int] = defaultdict(int)
    announced_batches: set[tuple[int, int]] = set()

    def emit(message: str) -> None:
        if session_id and supports_live_progress():
            append_event(session_id, message)
        else:
            print(message)

    def current_batch_key(exp_index: int) -> tuple[int, int] | None:
        batch_list = batch_order.get(exp_index, [])
        cursor = current_batch_index.get(exp_index, 0)
        if cursor >= len(batch_list):
            return None
        return (exp_index, batch_list[cursor])

    def advance_completed_batches() -> None:
        changed = True
        while changed:
            changed = False
            for exp_index in sorted(batch_order):
                batch_key = current_batch_key(exp_index)
                while batch_key is not None and not pending_by_batch[batch_key] and active_batch_jobs[batch_key] == 0:
                    groups = ",".join(batch_group_labels.get(batch_key, []))
                    emit(
                        f"[blocking-batch-done] exp={exp_index} "
                        f"blocking-batch={batch_key[1]} groups={groups}"
                    )
                    current_batch_index[exp_index] += 1
                    changed = True
                    batch_key = current_batch_key(exp_index)

    def launch(job: ScheduledGpuJob) -> subprocess.Popen[str]:
        env = os.environ.copy()
        env.update(worker_environment(job.assigned_gpu))
        if session_id and supports_live_progress():
            env["ResAssetPricing_PROGRESS_SESSION"] = session_id
            env["ResAssetPricing_PROGRESS_SLOT"] = str(job_slots[id(job)])
            env["ResAssetPricing_PROGRESS_GROUP_LABEL"] = job.progress_group
            env["ResAssetPricing_PROGRESS_ROW_LABEL"] = f"  {job.progress_item}"
            env["ResAssetPricing_PROGRESS_LABEL"] = f"device {job.assigned_gpu}"
        emit(f"[launch] blocking-batch={job.blocking_batch} gpu={job.assigned_gpu} {job.label}")
        return subprocess.Popen(job.command, cwd=str(project_root), env=env, text=True)

    while True:
        advance_completed_batches()
        launched = False

        for exp_index in sorted(batch_order):
            batch_key = current_batch_key(exp_index)
            if batch_key is None:
                continue
            if batch_key not in announced_batches:
                groups = ",".join(batch_group_labels.get(batch_key, []))
                emit(
                    f"[blocking-batch-open] exp={exp_index} "
                    f"blocking-batch={batch_key[1]} groups={groups} jobs={len(pending_by_batch[batch_key])}"
                )
                announced_batches.add(batch_key)
            pending = pending_by_batch[batch_key]
            index = 0
            while index < len(pending):
                job = pending[index]
                gpu = job.assigned_gpu
                if gpu not in devices:
                    index += 1
                    continue
                if active_counts[gpu] >= effective_gpu_total_limit(
                    gpu,
                    candidate_layer=job.layer,
                    gpu_limits=gpu_limits,
                    active_layer_limits=active_layer_limits,
                ):
                    index += 1
                    continue
                batch_gpu_key = (job.experiment_index, job.blocking_batch, gpu)
                if active_batch_counts[batch_gpu_key] >= batch_gpu_limits.get(batch_gpu_key, gpu_limits.get(gpu, 20)):
                    index += 1
                    continue
                process = launch(job)
                active[id(process)] = (process, job)
                active_counts[gpu] += 1
                active_layer_limits[gpu].append(depth_concurrency_limit(job.layer))
                active_batch_counts[batch_gpu_key] += 1
                active_batch_jobs[(job.experiment_index, job.blocking_batch)] += 1
                pending.pop(index)
                launched = True

        if not active and not launched:
            remaining = any(current_batch_key(exp_index) is not None for exp_index in batch_order)
            if remaining:
                raise RuntimeError(
                    "No schedulable device jobs remain; check per-device caps and blocking-batch assignments"
                )
            break

        time.sleep(0.2)
        for process_key, (process, job) in list(active.items()):
            status = process.poll()
            if status is None:
                continue
            del active[process_key]
            batch_gpu_key = (job.experiment_index, job.blocking_batch, job.assigned_gpu)
            active_counts[job.assigned_gpu] -= 1
            try:
                active_layer_limits[job.assigned_gpu].remove(depth_concurrency_limit(job.layer))
            except ValueError:
                pass
            active_batch_counts[batch_gpu_key] -= 1
            active_batch_jobs[(job.experiment_index, job.blocking_batch)] -= 1
            if status != 0:
                for other_process, _ in active.values():
                    other_process.terminate()
                raise subprocess.CalledProcessError(status, job.command)
            emit(f"[done] blocking-batch={job.blocking_batch} gpu={job.assigned_gpu} {job.label}")


def prepare_pretrained_checkpoints(payload: dict[str, Any], project_root: Path, *, dry_run: bool) -> None:
    """Extract only selected checkpoints, keeping trained models and results separate."""
    if not payload.get("use_pretrained"):
        return
    archive = project_root / "ResnetAssetPricing_pretrained_checkpoints.zip"
    selected: set[tuple[str, int]] = set()
    for experiment in payload["experiments"]:
        if not experiment.get("enabled"):
            continue
        entries = payload_layer_entries(experiment)
        if payload_runs_group_ablation_only(payload):
            entries = group_ablation_layers_with_anchor(experiment["name"], entries)
        for entry in entries:
            if not entry.get("enabled"):
                continue
            prefix = asset_prefix_for_experiment(experiment["name"], int(entry["layer"]))
            selected.update((prefix, int(cell["ensemble"])) for cell in entry["cells"] if cell.get("enabled"))
    if not selected:
        raise ValueError("Select at least one model and seed for pretrained prediction.")
    if dry_run:
        print(f"[dry-run] Extract selected pretrained checkpoints from {archive}: {sorted(selected)}")
        return
    if not archive.is_file():
        raise FileNotFoundError(f"Place the fully downloaded pretrained ZIP at {archive}, then rerun.")

    from config.base import DataConfig
    from types import SimpleNamespace
    import pandas as pd
    data = DataConfig(SimpleNamespace(device="cpu"))
    dates = pd.date_range(data.time_loop_start, data.time_loop_end, freq=data.time_loop_freq).strftime("%Y%m%d")
    required = {(prefix, date, ensemble) for prefix, ensemble in selected for date in dates}
    pattern = re.compile(r"asset/([^/]+)/ckpt/(\d{8})/ensemble_(\d+)/epoch_(\d+)\.pth")
    try:
        with zipfile.ZipFile(archive) as source:
            members = {}
            for info in source.infolist():
                match = pattern.fullmatch(info.filename)
                if match is None:
                    continue
                prefix, date, ensemble, epoch = match.groups()
                key = (prefix, date, int(ensemble))
                if key not in required:
                    continue
                if stat.S_ISLNK(info.external_attr >> 16):
                    raise ValueError(f"Checkpoint is a symbolic link: {info.filename}")
                if key not in members or int(epoch) > members[key][0]:
                    members[key] = (int(epoch), info)
            missing = required - members.keys()
            if missing:
                raise ValueError(f"Pretrained archive lacks {len(missing)} selected model/year/seed checkpoints; first: {min(missing)}")
            for index, (key, (_, info)) in enumerate(sorted(members.items()), start=1):
                prefix, date, ensemble = key
                target = project_root / "asset" / f"pretrained_{prefix}" / "ckpt" / date / f"ensemble_{ensemble}" / Path(info.filename).name
                if target.exists():
                    checksum = 0
                    with target.open("rb") as saved:
                        for block in iter(lambda: saved.read(1024 * 1024), b""):
                            checksum = zlib.crc32(block, checksum)
                    if target.stat().st_size != info.file_size or checksum != info.CRC:
                        raise ValueError(f"Existing pretrained checkpoint differs from the ZIP: {target}")
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".partial", delete=False) as staging:
                    temporary = Path(staging.name)
                    try:
                        with source.open(info) as checkpoint:
                            shutil.copyfileobj(checkpoint, staging)
                    except BaseException:
                        staging.close()
                        temporary.unlink(missing_ok=True)
                        raise
                try:
                    temporary.replace(target)
                finally:
                    temporary.unlink(missing_ok=True)
                if index == 1 or index == len(members) or index % 50 == 0:
                    print(f"[pretrained] Prepared {index}/{len(members)} selected checkpoints", flush=True)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"Pretrained ZIP is incomplete or damaged: {archive}. Finish downloading it before retrying.") from exc


def execute_payload(
    payload: dict[str, Any],
    *,
    project_root: Path,
    python_bin: str,
    dry_run: bool,
) -> None:
    prepare_pretrained_checkpoints(payload, project_root, dry_run=dry_run)
    current_devices = detect_devices()
    previous_devices = [str(device) for device in payload.get("devices", current_devices)]
    if previous_devices == ["cpu"]:
        current_devices = ["cpu"]
    elif previous_devices and all(device in current_devices for device in previous_devices):
        # Preserve a saved plan's GPU allocation instead of expanding to every visible GPU.
        current_devices = previous_devices
    payload = copy.deepcopy(payload)
    payload["devices"] = current_devices
    payload["gpu_slots"] = resolve_payload_gpu_slots(payload.get("gpu_slots"), current_devices, previous_devices)
    for experiment in payload["experiments"]:
        entries = experiment.get("layers", [])
        if entries and isinstance(entries[0], int):
            entries = [experiment]
        for entry in entries:
            for cell in entry.get("cells", []):
                mapped = resolve_payload_gpu_list(cell.get("gpus", []), current_devices, previous_devices)
                if experiment.get("enabled") and entry.get("enabled", True) and cell.get("enabled") and not mapped:
                    raise ValueError("A selected job has no available device; reassign devices in the TUI.")
                cell["gpus"] = mapped
    print(device_message(current_devices))
    if not dry_run:
        run_subprocess([python_bin, "-m", "utils.check_runtime", "--project-root", str(project_root)],
                       cwd=project_root)
    main_py = project_root / "main.py"
    prepare_data_py = project_root / "utils" / "prepare_data.py"
    use_presplit = ensure_runtime_ready(
        project_root=project_root,
        python_bin=python_bin,
        main_py=main_py,
        prepare_data_py=prepare_data_py,
        dry_run=dry_run,
        group_ablation_only=payload_runs_group_ablation_only(payload),
    )
    jobs_by_blocking_batch, foreground_by_blocking_batch = build_execution_jobs(
        payload,
        project_root=project_root,
        python_bin=python_bin,
        main_py=main_py,
        use_presplit=use_presplit,
    )

    session_id: str | None = None
    gpu_slots = normalized_gpu_slots(payload.get("gpu_slots"), payload["devices"])
    if payload_runs_group_ablation_only(payload):
        gpu_slots = {device: device_slots(device, GROUP_ABLATION_GPU_CONCURRENCY) for device in payload["devices"]}
    if not dry_run and supports_live_progress():
        session_id = f"tui_run_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
        init_session(session_id, 0, progress_title(payload))

    status_message = "ResAssetPricing launcher finished."
    try:
        execute_gpu_jobs_by_blocking_batch(
            jobs_by_blocking_batch,
            payload=payload,
            foreground_by_blocking_batch=foreground_by_blocking_batch,
            project_root=project_root,
            devices=payload["devices"],
            gpu_slots=gpu_slots,
            session_id=session_id,
            dry_run=dry_run,
        )
        if payload_runs_group_ablation_only(payload):
            print("[skip] group-ablation-only does not refresh report sources")
        elif payload.get("use_pretrained"):
            print("[pretrained] Predictions and requested evaluation are saved under asset/pretrained_*.")
        elif payload_runs_econ(payload):
            refresh_report_sources(
                project_root=project_root,
                python_bin=python_bin,
                ensemble_ids=[int(ensemble) for ensemble in payload["ensemble_ids"]],
                dry_run=dry_run,
            )
        else:
            print("[skip] econ and report source refresh disabled by plan")
    except BaseException as exc:
        status_message = "ResAssetPricing launcher aborted."
        if session_id and supports_live_progress():
            append_event(session_id, f"[error] {type(exc).__name__}: {exc}")
        raise
    finally:
        if session_id is not None:
            finalize_session(session_id, status_message)


class LauncherUI:
    def __init__(self, stdscr, state: dict[str, Any], *, dry_run: bool):
        self.stdscr = stdscr
        self.state = state
        self.dry_run = dry_run
        self.row_index = 0
        self.col_index = -1
        self.scroll_top = 0
        self.message = device_message(state["devices"])
        self.result: dict[str, Any] | None = None
        self.gpu_utils = {device: None for device in self.state["devices"]}
        self.last_gpu_utils_refresh = 0.0
        self.last_plan_path: Path | None = latest_saved_plan_path()
        self.check_mark = ASCII_CHECK_MARK if should_use_ascii_tui_symbols(self.stdscr) else CHECK_MARK
        self.stdscr.timeout(UI_POLL_INTERVAL_MS)

    def visible_rows(self) -> list[tuple[str, str, int | None]]:
        rows: list[tuple[str, str, int | None]] = []
        for name in EXPERIMENT_ORDER:
            rows.append(("experiment", name, None))
            experiment = self.state["experiments"][name]
            if not experiment_is_enabled(experiment):
                continue
            for layer in resolved_layers_for_state(name, experiment):
                rows.append(("layer", name, layer))
        return rows

    def current_entry(self) -> tuple[str, str, int | None]:
        rows = self.visible_rows()
        self.row_index = max(0, min(self.row_index, len(rows) - 1))
        return rows[self.row_index]

    def current_experiment_name(self) -> str:
        _, name, _ = self.current_entry()
        return name

    def current_row(self) -> ExperimentState:
        return self.state["experiments"][self.current_experiment_name()]

    def current_layer(self) -> int | None:
        _, _, layer = self.current_entry()
        return layer

    def current_layer_state(self) -> LayerState | None:
        layer = self.current_layer()
        if layer is None:
            return None
        return self.current_row().layer_states[layer]

    def current_ensemble(self) -> int | None:
        if self.col_index < 0:
            return None
        return self.state["ensemble_ids"][self.col_index]

    def current_cell(self) -> CellState | None:
        ensemble = self.current_ensemble()
        layer_state = self.current_layer_state()
        if ensemble is None or layer_state is None:
            return None
        return layer_state.cells[ensemble]

    def current_scope_label(self) -> str:
        name = self.current_experiment_name()
        layer = self.current_layer()
        ensemble = self.current_ensemble()
        layer_label = f" {self.format_layer(layer)}" if layer is not None else ""
        ensemble_label = f" ensemble {ensemble}" if ensemble is not None else ""
        return f"{name}{layer_label}{ensemble_label}"

    def target_cells(self) -> list[CellState]:
        row = self.current_row()
        layer = self.current_layer()
        ensemble = self.current_ensemble()
        if layer is None:
            layers = resolved_layers_for_state(self.current_experiment_name(), row)
            if ensemble is None:
                return [row.layer_states[layer_id].cells[item] for layer_id in layers for item in self.state["ensemble_ids"]]
            return [row.layer_states[layer_id].cells[ensemble] for layer_id in layers]
        if ensemble is None:
            return [row.layer_states[layer].cells[item] for item in self.state["ensemble_ids"]]
        return [row.layer_states[layer].cells[ensemble]]

    def target_layer_states(self) -> list[LayerState]:
        row = self.current_row()
        layer = self.current_layer()
        if layer is None:
            return [row.layer_states[layer_id] for layer_id in resolved_layers_for_state(self.current_experiment_name(), row)]
        return [row.layer_states[layer]]

    def group_ensemble_display(self) -> str:
        layer_states = self.target_layer_states()
        enabled_sets = [
            tuple(ensemble for ensemble in self.state["ensemble_ids"] if layer_state.cells[ensemble].enabled)
            for layer_state in layer_states
        ]
        if enabled_sets and all(enabled_set == enabled_sets[0] for enabled_set in enabled_sets):
            if not enabled_sets[0]:
                return "-"
            return ",".join(str(ensemble) for ensemble in enabled_sets[0])
        return "mixed"

    def set_message(self, text: str) -> None:
        self.message = text

    def clear_message(self) -> None:
        self.message = ""

    def experiment_row_indices(self) -> list[int]:
        return [index for index, (kind, _, _) in enumerate(self.visible_rows()) if kind == "experiment"]

    def jump_experiment(self, direction: int) -> None:
        indices = self.experiment_row_indices()
        if not indices:
            return
        current = self.row_index
        if direction > 0:
            for index in indices:
                if index > current:
                    self.row_index = index
                    return
            self.row_index = indices[-1]
            return
        for index in reversed(indices):
            if index < current:
                self.row_index = index
                return
        self.row_index = indices[0]

    def maybe_refresh_gpu_utils(self, *, force: bool = False) -> None:
        devices = self.state["devices"]
        if not devices:
            self.gpu_utils = {}
            self.last_gpu_utils_refresh = time.monotonic()
            return

        now = time.monotonic()
        if not force and now - self.last_gpu_utils_refresh < GPU_UTIL_REFRESH_SECONDS:
            return

        self.gpu_utils = detect_gpu_utils(devices)
        self.last_gpu_utils_refresh = now

    def gpu_util_line(self) -> str:
        devices = self.state["devices"]
        if devices in (["cpu"], ["mps"]):
            return f"Device: {devices[0].upper()}"
        if not devices:
            return "GPU util: none detected"

        self.maybe_refresh_gpu_utils()
        parts: list[str] = []
        missing = 0
        for position, device in enumerate(devices):
            util = self.gpu_utils.get(device)
            if util is None:
                missing += 1
                parts.append(f"{position}=?")
            else:
                parts.append(f"{position}={util}%")

        prefix = "GPU util:"
        if missing == len(devices):
            prefix = "GPU util: unavailable"
        return f"{prefix} {' | '.join(parts)}"

    def prompt(self, text: str) -> str | None:
        height, width = self.stdscr.getmaxyx()
        prompt_text = text[: max(width - 1, 1)]
        curses.echo()
        safe_curs_set(1)
        self.stdscr.timeout(-1)
        self.stdscr.move(height - 1, 0)
        self.stdscr.clrtoeol()
        safe_addnstr(self.stdscr, height - 1, 0, prompt_text, width - 1)
        self.stdscr.refresh()
        try:
            raw = self.stdscr.getstr(height - 1, len(prompt_text), max(width - len(prompt_text) - 1, 1))
            value = raw.decode("utf-8").strip()
            return value or None
        finally:
            curses.noecho()
            safe_curs_set(0)
            self.stdscr.timeout(UI_POLL_INTERVAL_MS)

    def toggle_current(self) -> None:
        name = self.current_experiment_name()
        row = self.current_row()
        scope_label = self.current_scope_label()
        has_ensemble_column = self.current_ensemble() is not None
        targets = self.target_cells()
        next_value = not all(cell.enabled for cell in targets)
        for cell in targets:
            cell.enabled = next_value
        if not has_ensemble_column:
            auto_plan_experiment(
                name,
                row,
                self.state["ensemble_ids"],
                self.state["devices"],
            )
        else:
            auto_plan_blocking_batches_for_experiment(
                name,
                row,
                self.state["ensemble_ids"],
            )
        normalize_enabled_blocking_batches(self.state)
        self.set_message(f"{scope_label} enabled={next_value}")

    def edit_layers(self) -> None:
        name = self.current_experiment_name()
        row = self.current_row()
        value = self.prompt(f"Layers for {name} [{row.layers_spec}]> ")
        if value is None:
            return
        layers = parse_integer_spec(value, "layer")
        validate_layers(name, layers)
        row.layers_spec = value
        sync_layer_states(name, row, self.state["ensemble_ids"], self.state["devices"])
        auto_plan_experiment(name, row, self.state["ensemble_ids"], self.state["devices"])
        normalize_enabled_blocking_batches(self.state)
        self.set_message(f"{name} layers={value}")

    def edit_seeded_family_layers(self) -> None:
        current_name = self.current_experiment_name()
        family_members = seeded_family_members(current_name)
        family_kind, _ = parse_seeded_experiment_name(current_name)
        value = self.prompt(f"Layers for all {family_kind} seeds [e.g. 1-{MAX_HIDDEN_LAYER_COUNT}]> ")
        if value is None:
            return
        start, end = parse_layer_range_shortcut(value)
        updated: list[str] = []
        disabled: list[str] = []
        for name in family_members:
            row = self.state["experiments"][name]
            _, seed = parse_seeded_experiment_name(name)
            layer_start = max(start, seed)
            layer_end = min(end, MAX_HIDDEN_LAYER_COUNT)
            if layer_start <= layer_end:
                row.layers_spec = f"{layer_start}:{layer_end}" if layer_start != layer_end else str(layer_start)
                sync_layer_states(name, row, self.state["ensemble_ids"], self.state["devices"])
                for layer_state in row.layer_states.values():
                    for cell in layer_state.cells.values():
                        cell.enabled = True
                auto_plan_experiment(name, row, self.state["ensemble_ids"], self.state["devices"])
                updated.append(f"{name}={row.layers_spec}")
                continue

            row.layers_spec = str(seed)
            sync_layer_states(name, row, self.state["ensemble_ids"], self.state["devices"])
            for layer_state in row.layer_states.values():
                for cell in layer_state.cells.values():
                    cell.enabled = False
            auto_plan_experiment(name, row, self.state["ensemble_ids"], self.state["devices"])
            disabled.append(name)

        normalize_enabled_blocking_batches(self.state)
        summary = str(start) if start == end else f"{start}-{end}"
        if disabled:
            self.set_message(f"{family_kind} seed layers={summary}; disabled no-overlap: {','.join(disabled)}")
        else:
            self.set_message(f"{family_kind} seed layers={summary}: {', '.join(updated)}")

    def edit_row_ensembles(self) -> None:
        scope = self.current_scope_label()
        value = self.prompt(f"Enable ensembles for {scope} [{self.group_ensemble_display()}]> ")
        if value is None:
            return
        selected = parse_integer_spec(value, "ensemble", allow_desc=False)
        missing = [ensemble for ensemble in selected if ensemble not in self.state["ensemble_ids"]]
        if missing:
            raise ValueError(f"Ensembles {missing} are not in the current matrix columns")
        chosen = set(selected)
        for layer_state in self.target_layer_states():
            for ensemble in self.state["ensemble_ids"]:
                layer_state.cells[ensemble].enabled = ensemble in chosen
        if self.current_layer() is None:
            auto_plan_experiment(
                self.current_experiment_name(),
                self.current_row(),
                self.state["ensemble_ids"],
                self.state["devices"],
            )
        else:
            auto_plan_blocking_batches_for_experiment(
                self.current_experiment_name(),
                self.current_row(),
                self.state["ensemble_ids"],
            )
        normalize_enabled_blocking_batches(self.state)
        self.set_message(f"{scope} ensembles={value}")

    def edit_global_ensembles(self) -> None:
        current = ",".join(str(value) for value in self.state["ensemble_ids"])
        value = self.prompt(f"Global ensemble columns [{current}]> ")
        if value is None:
            return
        ensemble_ids = parse_integer_spec(value, "ensemble", allow_desc=False)
        if not ensemble_ids:
            raise ValueError("Global ensemble columns cannot be empty")
        sync_ensemble_columns(self.state, ensemble_ids)
        self.col_index = min(self.col_index, len(self.state["ensemble_ids"]) - 1)
        self.set_message(f"Matrix columns={value}")

    def edit_gpu_slots(self) -> None:
        if self.state["devices"] in (["cpu"], ["mps"]):
            self.set_message(device_message(self.state["devices"]))
            return
        current = gpu_slots_summary(self.state.get("gpu_slots", {}), self.state["devices"])
        value = self.prompt(f"Device concurrency [pos=cap,... or int for all] [{current}]> ")
        if value is None:
            return
        self.state["gpu_slots"] = parse_gpu_slot_spec(value, self.state["devices"], self.state.get("gpu_slots", {}))
        self.set_message(f"device total caps={gpu_slots_summary(self.state['gpu_slots'], self.state['devices'])}")

    def assign_gpus_to_layer_state(self, layer_state: LayerState, selected: list[str]) -> None:
        assignment = balanced_gpu_assignment(self.state["ensemble_ids"], selected)
        for ensemble in self.state["ensemble_ids"]:
            layer_state.cells[ensemble].gpus = [assignment[ensemble]]

    def edit_gpus(self) -> None:
        devices = self.state["devices"]
        available = available_gpu_positions(devices)
        value = self.prompt(f"Device preference (single device, all, or positions like 0,1 for balanced bulk assign) [available {available}]> ")
        if value is None:
            return
        selected = parse_gpu_spec(value, devices)
        ensemble = self.current_ensemble()
        if ensemble is not None:
            if len(selected) != 1:
                raise ValueError("A selected ensemble can only have one preferred device")
            for cell in self.target_cells():
                cell.gpus = [selected[0]]
            auto_plan_blocking_batches_for_experiment(
                self.current_experiment_name(),
                self.current_row(),
                self.state["ensemble_ids"],
            )
            normalize_enabled_blocking_batches(self.state)
            self.set_message(f"{self.current_scope_label()} device-pref={device_positions(selected, devices)}")
            return
        if self.current_layer() is not None:
            self.assign_gpus_to_layer_state(self.current_layer_state(), selected)
            auto_plan_blocking_batches_for_experiment(
                self.current_experiment_name(),
                self.current_row(),
                self.state["ensemble_ids"],
            )
            normalize_enabled_blocking_batches(self.state)
            self.set_message(f"{self.current_scope_label()} device-pref-plan={device_positions(selected, devices)}")
            return
        auto_plan_experiment(self.current_experiment_name(), self.current_row(), self.state["ensemble_ids"], selected)
        normalize_enabled_blocking_batches(self.state)
        self.set_message(f"{self.current_scope_label()} device-pref-plan={device_positions(selected, devices)}")

    def edit_blocking_batch(self) -> None:
        value = self.prompt("Blocking batch integer (auto-renumbered globally)> ")
        if value is None:
            return
        blocking_batch = int(value)
        if blocking_batch <= 0:
            raise ValueError("Blocking batch must be >= 1")
        targets = self.target_cells()
        for cell in targets:
            cell.blocking_batch = blocking_batch
        normalize_enabled_blocking_batches(self.state)
        enabled_batches = sorted({cell.blocking_batch for cell in targets if cell.enabled})
        if len(enabled_batches) == 1:
            self.set_message(f"{self.current_scope_label()} blocking-batch={enabled_batches[0]}")
            return
        if enabled_batches:
            batch_text = ",".join(str(batch) for batch in enabled_batches)
            self.set_message(f"{self.current_scope_label()} blocking-batches={batch_text}")
            return
        self.set_message(f"{self.current_scope_label()} blocking-batch order updated")

    def reset_row(self) -> None:
        name = self.current_experiment_name()
        exp_def = EXPERIMENTS[name]
        row = self.current_row()
        layer = self.current_layer()
        if layer is None:
            row.layers_spec = exp_def.default_layers
            row.layer_states = {}
            sync_layer_states(name, row, self.state["ensemble_ids"], self.state["devices"])
            auto_plan_experiment(name, row, self.state["ensemble_ids"], self.state["devices"])
            normalize_enabled_blocking_batches(self.state)
            self.set_message(f"{name} reset to defaults")
            return
        row.layer_states[layer] = make_default_layer_state(exp_def, self.state["ensemble_ids"], self.state["devices"])
        auto_plan_experiment(name, row, self.state["ensemble_ids"], self.state["devices"])
        normalize_enabled_blocking_batches(self.state)
        self.set_message(f"{name} {self.format_layer(layer)} reset to defaults")

    def enable_all_row(self) -> None:
        for cell in self.target_cells():
            cell.enabled = True
        auto_plan_experiment(
            self.current_experiment_name(),
            self.current_row(),
            self.state["ensemble_ids"],
            self.state["devices"],
        )
        normalize_enabled_blocking_batches(self.state)
        self.set_message(f"{self.current_scope_label()} enabled")

    def clear_current(self) -> None:
        name = self.current_experiment_name()
        row = self.current_row()
        scope_label = self.current_scope_label()
        has_ensemble_column = self.current_ensemble() is not None
        for cell in self.target_cells():
            cell.enabled = False
        if not has_ensemble_column:
            auto_plan_experiment(
                name,
                row,
                self.state["ensemble_ids"],
                self.state["devices"],
            )
        else:
            auto_plan_blocking_batches_for_experiment(
                name,
                row,
                self.state["ensemble_ids"],
            )
        normalize_enabled_blocking_batches(self.state)
        self.set_message(f"{scope_label} cleared")

    def save_plan(self) -> None:
        payload = build_plan_payload(self.state)
        default_path = Path(tempfile.gettempdir()) / f"resassetpricing_run_plan_{time.strftime('%Y%m%d_%H%M%S')}.json"
        value = self.prompt(f"Save plan path [{default_path}]> ")
        path = Path(value) if value else default_path
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")
        self.last_plan_path = path
        self.set_message(f"Plan saved to {path}")

    def load_plan(self) -> None:
        default_path = self.last_plan_path or latest_saved_plan_path()
        prompt = f"Load plan path [{default_path}]> " if default_path is not None else "Load plan path> "
        value = self.prompt(prompt)
        path = Path(value) if value else default_path
        if path is None:
            raise ValueError("No saved plan path was provided")
        payload = load_payload(path)
        self.state = state_from_payload(payload, self.state["devices"])
        self.gpu_utils = {device: None for device in self.state["devices"]}
        self.last_gpu_utils_refresh = 0.0
        self.last_plan_path = path
        self.row_index = 0
        self.col_index = -1
        self.scroll_top = 0
        self.set_message(f"Plan loaded from {path}")

    def queue_run(self) -> None:
        payload = build_plan_payload(self.state)
        default_pretrained = "y" if self.state.get("use_pretrained") else "n"
        value = self.prompt(f"Use pretrained checkpoints and skip training? [y/N, default {default_pretrained}]> ")
        choice = (value or default_pretrained).strip().lower()
        if choice not in {"y", "yes", "n", "no"}:
            raise ValueError("Enter y or n for pretrained checkpoints")
        payload["use_pretrained"] = self.state["use_pretrained"] = choice in {"y", "yes"}
        value = self.prompt(
            "Run mode [1 predict+econ, 2 predict only, 3 group-ablation] [1]> "
            if payload["use_pretrained"] else
            "Run mode [1 train+predict+econ, 2 train+predict, 3 group-ablation+shallow-anchors] [1]> "
        )
        mode = "1" if value is None or not value.strip() else value.strip()
        if mode not in {"1", "2", "3"}:
            raise ValueError("Run mode must be 1, 2, or 3")
        payload["run_econ"] = mode == "1"
        payload["group_ablation_only"] = mode == "3"
        self.result = {"action": "run", "payload": payload}

    def gpu_tag(self, gpus: list[str]) -> str:
        if len(gpus) == 1 and gpus[0] in self.state["devices"]:
            return str(self.state["devices"].index(gpus[0]))
        if not gpus:
            return "-"
        return "+"

    def format_cell_group(self, cells: list[CellState]) -> str:
        enabled_cells = [cell for cell in cells if cell.enabled]
        if not enabled_cells:
            return "  .  "
        blocking_batch_tags = {cell.blocking_batch for cell in enabled_cells}
        gpu_tags = {self.gpu_tag(cell.gpus) for cell in enabled_cells}
        if len(enabled_cells) == len(cells) and len(blocking_batch_tags) == 1 and len(gpu_tags) == 1:
            blocking_batch = next(iter(blocking_batch_tags))
            gpu_tag = next(iter(gpu_tags))
            return f"{blocking_batch:02d}:{gpu_tag}"[:5].ljust(5)
        return " mix "

    def format_summary_cell(self, cells: list[CellState]) -> str:
        enabled_count = sum(1 for cell in cells if cell.enabled)
        if enabled_count == 0:
            return "  .  "
        return f"{enabled_count:02d}L".ljust(5)

    def format_layer(self, layer: int) -> str:
        return f"L{layer:02d}"

    def row_meta_text(self, kind: str, name: str, layer: int | None) -> str:
        row = self.state["experiments"][name]
        marker = self.check_mark if experiment_is_enabled(row) else " "
        if kind == "experiment":
            layers = resolved_layers_for_state(name, row)
            enabled_layers = sum(1 for layer_id in layers if layer_is_enabled(row.layer_states[layer_id]))
            enabled_cells = sum(
                1
                for layer_id in layers
                for ensemble in self.state["ensemble_ids"]
                if row.layer_states[layer_id].cells[ensemble].enabled
            )
            return f"[{marker}] {name:<18} {row.layers_spec} | {enabled_layers} Layers {enabled_cells} Ensembles"

        assert layer is not None
        layer_marker = self.check_mark if layer_is_enabled(row.layer_states[layer]) else " "
        return f"  [{layer_marker}] {self.format_layer(layer):<18}"

    def selected_detail(self) -> str:
        name = self.current_experiment_name()
        row = self.current_row()
        exp_def = EXPERIMENTS[name]
        layers = resolved_layers_for_state(name, row)
        if self.current_layer() is None:
            enabled_layers = sum(1 for layer in layers if layer_is_enabled(row.layer_states[layer]))
            if self.col_index < 0:
                enabled_cells = sum(
                    1
                    for layer in layers
                    for ensemble in self.state["ensemble_ids"]
                    if row.layer_states[layer].cells[ensemble].enabled
                )
                total_cells = len(layers) * len(self.state["ensemble_ids"])
                return (
                    f"{name} | enabled layers={enabled_layers}/{len(layers)} | layers={row.layers_spec} | "
                    f"enabled cells={enabled_cells}/{total_cells} | device caps={gpu_slots_summary(self.state.get('gpu_slots', {}), self.state['devices'])} | role={exp_def.role}"
                )

            ensemble = self.current_ensemble()
            assert ensemble is not None
            enabled_count = sum(1 for layer in layers if row.layer_states[layer].cells[ensemble].enabled)
            return (
                f"{name} e{ensemble:02d} | layers={row.layers_spec} | enabled layers={enabled_layers}/{len(layers)} | "
                f"enabled cells={enabled_count}/{len(layers)} | role={exp_def.role}"
            )

        layer = self.current_layer()
        assert layer is not None
        layer_state = row.layer_states[layer]
        ensemble = self.current_ensemble()
        if ensemble is None:
            enabled_count = sum(1 for cell in layer_state.cells.values() if cell.enabled)
            return (
                f"{name} {self.format_layer(layer)} | layer={'on' if layer_is_enabled(layer_state) else 'off'} | "
                f"ensembles={enabled_count}/{len(layer_state.cells)} | valid={exp_def.valid_layers} | role={exp_def.role}"
            )

        cell = layer_state.cells[ensemble]
        return (
            f"{name} {self.format_layer(layer)} e{ensemble:02d} | "
            f"cell={'on' if cell.enabled else 'off'} | blocking-batch={cell.blocking_batch} | "
            f"device-pref={device_positions(cell.gpus, self.state['devices'])} | layers={row.layers_spec}"
        )

    def visible_table_rows(self, height: int) -> int:
        rows = self.visible_rows()
        _, _, _, table_rows = self.layout_for_scroll(height, rows, self.scroll_top)
        return table_rows

    def layout_for_scroll(self, height: int, rows: list[tuple[str, str, int | None]], scroll_top: int) -> tuple[int, str | None, int, int]:
        intro_rows = 3 if scroll_top == 0 else 0
        sticky_experiment = self.sticky_experiment_name(rows, scroll_top)
        sticky_rows = 1 if sticky_experiment is not None else 0
        footer_rows = 1
        base_row = 1 + intro_rows + sticky_rows
        table_rows = max(height - base_row - footer_rows, 1)
        return intro_rows, sticky_experiment, base_row, table_rows

    def sticky_experiment_name(self, rows: list[tuple[str, str, int | None]], scroll_top: int) -> str | None:
        if scroll_top <= 0 or scroll_top >= len(rows):
            return None
        kind, name, _ = rows[scroll_top]
        if kind != "layer":
            return None
        return name

    def legend_meta_text(self) -> str:
        return "[ ] Experiment       layers | Count"

    def legend_layer_off_text(self) -> str:
        return "  [ ] NotRun"

    def legend_layer_on_text(self) -> str:
        return f"  [{self.check_mark}] Run"

    def footer_text(self) -> str:
        return self.message or "Arrows move | [ ] jump experiment | Space toggle | d seed layers | x reset | s save | f load | r run-mode prompt | q quit"

    def render(self) -> None:
        self.stdscr.erase()
        height, width = self.stdscr.getmaxyx()
        safe_curs_set(0)

        rows = self.visible_rows()
        self.row_index = max(0, min(self.row_index, len(rows) - 1))
        cell_width = 6
        min_meta_width = 24
        available_meta_width = max(width - len(self.state["ensemble_ids"]) * cell_width - 1, min_meta_width)
        content_width = max(
            len("Experiment / Layer"),
            max(len(self.row_meta_text(kind, name, layer)) for kind, name, layer in rows),
            len(self.legend_meta_text()),
            len(self.legend_layer_off_text()),
            len(self.legend_layer_on_text()),
        )
        meta_width = min(max(content_width + 2, min_meta_width), available_meta_width)
        header = "Experiment / Layer".ljust(meta_width)
        for ensemble in self.state["ensemble_ids"]:
            header += f"E{ensemble:02d}".center(cell_width)
        safe_addnstr(self.stdscr, 0, 0, header, width - 1, curses.A_BOLD)

        max_scroll = max(0, len(rows) - 1)
        self.scroll_top = min(self.scroll_top, max_scroll)
        for _ in range(3):
            _, _, _, table_rows = self.layout_for_scroll(height, rows, self.scroll_top)
            max_scroll = max(0, len(rows) - table_rows)
            self.scroll_top = min(self.scroll_top, max_scroll)
            if self.row_index < self.scroll_top:
                self.scroll_top = self.row_index
            elif self.row_index >= self.scroll_top + table_rows:
                self.scroll_top = self.row_index - table_rows + 1

        intro_rows, sticky_experiment, base_row, table_rows = self.layout_for_scroll(height, rows, self.scroll_top)
        cursor_y = 1
        if intro_rows:
            safe_addnstr(self.stdscr, cursor_y, 0, self.legend_meta_text().ljust(meta_width), width - 1)
            cursor_y += 1
            safe_addnstr(self.stdscr, cursor_y, 0, self.legend_layer_off_text().ljust(meta_width), meta_width)
            safe_addnstr(self.stdscr, cursor_y, meta_width, "job_seq:gpu_pref", width - meta_width - 1)
            cursor_y += 1
            safe_addnstr(self.stdscr, cursor_y, 0, self.legend_layer_on_text().ljust(meta_width), meta_width)
            cursor_y += 1

        if sticky_experiment is not None:
            sticky_meta = self.row_meta_text("experiment", sticky_experiment, None)
            safe_addnstr(self.stdscr, cursor_y, 0, sticky_meta.ljust(meta_width), meta_width, curses.A_BOLD)
            cursor_y += 1

        for row_offset, (kind, name, layer) in enumerate(rows[self.scroll_top : self.scroll_top + table_rows]):
            y = cursor_y + row_offset
            row = self.state["experiments"][name]
            meta = self.row_meta_text(kind, name, layer)
            absolute_row = self.scroll_top + row_offset
            meta_attr = curses.A_REVERSE if self.row_index == absolute_row and self.col_index < 0 else curses.A_NORMAL
            safe_addnstr(self.stdscr, y, 0, meta.ljust(meta_width), meta_width, meta_attr)

            x = meta_width
            for col_offset, ensemble in enumerate(self.state["ensemble_ids"]):
                if kind == "experiment":
                    cell_text = ""
                else:
                    assert layer is not None
                    cell_text = self.format_cell_group([row.layer_states[layer].cells[ensemble]])
                attr = curses.A_REVERSE if self.row_index == absolute_row and self.col_index == col_offset else curses.A_NORMAL
                safe_addnstr(self.stdscr, y, x, cell_text.center(cell_width), cell_width, attr)
                x += cell_width

        safe_addnstr(self.stdscr, height - 1, 0, self.footer_text(), width - 1)
        self.stdscr.refresh()

    def run(self) -> dict[str, Any] | None:
        while self.result is None:
            self.render()
            key = self.stdscr.getch()
            try:
                if key == -1:
                    continue
                if key == curses.KEY_UP:
                    self.clear_message()
                    self.row_index = max(0, self.row_index - 1)
                elif key == curses.KEY_DOWN:
                    self.clear_message()
                    self.row_index = min(len(self.visible_rows()) - 1, self.row_index + 1)
                elif key == curses.KEY_LEFT:
                    self.clear_message()
                    self.col_index = max(-1, self.col_index - 1)
                elif key == curses.KEY_RIGHT:
                    self.clear_message()
                    self.col_index = min(len(self.state["ensemble_ids"]) - 1, self.col_index + 1)
                elif key == curses.KEY_PPAGE:
                    self.clear_message()
                    self.row_index = max(0, self.row_index - self.visible_table_rows(self.stdscr.getmaxyx()[0]))
                elif key == curses.KEY_NPAGE:
                    self.clear_message()
                    self.row_index = min(
                        len(self.visible_rows()) - 1,
                        self.row_index + self.visible_table_rows(self.stdscr.getmaxyx()[0]),
                    )
                elif key == ord(" "):
                    self.toggle_current()
                elif key in (10, 13, curses.KEY_ENTER):
                    self.set_message("Press r to choose a run mode. Space toggles selection; q quits.")
                elif 0 <= key <= 255:
                    ch = chr(key).lower()
                    if ch == "q":
                        break
                    elif ch == "l":
                        self.edit_layers()
                    elif ch == "m":
                        self.edit_row_ensembles()
                    elif ch == "g":
                        self.edit_gpus()
                    elif ch == "o":
                        self.edit_blocking_batch()
                    elif ch == "p":
                        self.edit_gpu_slots()
                    elif ch == "d":
                        self.edit_seeded_family_layers()
                    elif ch == "x":
                        self.reset_row()
                    elif ch == "a":
                        self.enable_all_row()
                    elif ch == "c":
                        self.clear_current()
                    elif ch == "e":
                        self.edit_global_ensembles()
                    elif ch == "s":
                        self.save_plan()
                    elif ch == "f":
                        self.load_plan()
                    elif ch == "r":
                        self.queue_run()
                    elif ch == "[":
                        self.clear_message()
                        self.jump_experiment(-1)
                    elif ch == "]":
                        self.clear_message()
                        self.jump_experiment(1)
                    else:
                        self.set_message(f"Unknown key: {key}")
                else:
                    self.set_message(f"Unknown key: {key}")
            except Exception as exc:
                self.set_message(str(exc))
        return self.result


def run_interactive(*, dry_run: bool) -> dict[str, Any] | None:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise RuntimeError("The TUI needs an interactive terminal. Use an SSH terminal (ssh -t), docker run -it, or a saved --plan.")
    devices = detect_devices()
    state = default_state(devices)

    def _wrapped(stdscr):
        try:
            curses.use_default_colors()
        except curses.error:
            pass
        stdscr.keypad(True)
        return LauncherUI(stdscr, state, dry_run=dry_run).run()

    return curses.wrapper(_wrapped)


def load_payload(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid plan JSON: {path}") from exc
    if not isinstance(payload, dict) or "experiments" not in payload:
        raise ValueError(f"Invalid plan file: {path}")
    return payload


def latest_saved_plan_path() -> Path | None:
    candidates = sorted(Path(tempfile.gettempdir()).glob("resassetpricing_run_plan_*.json"))
    if not candidates:
        return None
    return candidates[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    interactive = subparsers.add_parser("interactive")
    interactive.add_argument("--dry-run", action="store_true", default=False)
    interactive.add_argument("--project-root", default=str(PROJECT_ROOT))
    interactive.add_argument("--python-bin", default=sys.executable)

    execute = subparsers.add_parser("execute")
    execute.add_argument("--plan", required=True)
    execute.add_argument("--dry-run", action="store_true", default=False)
    execute.add_argument("--project-root", default=str(PROJECT_ROOT))
    execute.add_argument("--python-bin", default=sys.executable)
    execute.add_argument("--no-econ", action="store_true", default=False)

    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        project_root = Path(args.project_root).resolve()
        python_bin = args.python_bin

        if args.command == "interactive":
            result = run_interactive(dry_run=args.dry_run)
            if not result:
                return 0
            execute_payload(
                result["payload"],
                project_root=project_root,
                python_bin=python_bin,
                dry_run=args.dry_run,
            )
            return 0

        payload = load_payload(Path(args.plan).resolve())
        if args.no_econ:
            payload["run_econ"] = False
        execute_payload(
            payload,
            project_root=project_root,
            python_bin=python_bin,
            dry_run=args.dry_run,
        )
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
