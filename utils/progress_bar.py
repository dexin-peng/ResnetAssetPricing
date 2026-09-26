import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None
try:
    import msvcrt
except ImportError:  # pragma: no cover
    msvcrt = None


RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
BLUE = "\033[34m"
GPU_UTIL_CACHE_SECONDS = 1.0


def _supports_live_terminal() -> bool:
    return sys.stdout.isatty() and os.environ.get("TERM", "dumb") != "dumb"


def _session_root(root: str | None = None) -> Path:
    raw = root or os.environ.get("ResAssetPricing_PROGRESS_ROOT") or str(Path(tempfile.gettempdir()) / "eapvml_progress")
    return Path(raw)


def _truncate(text: str, limit: int) -> str:
    text = str(text or "")
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return text[: limit - 3] + "..."


def _bar(fraction: float, width: int) -> str:
    width = max(width, 8)
    fraction = min(max(fraction, 0.0), 1.0)
    filled = min(width, int(round(fraction * width)))
    return "#" * filled + "-" * (width - filled)


def _style_for_status(status: str) -> str:
    status = (status or "").lower()
    if status == "done":
        return GREEN
    if status == "skipped":
        return YELLOW
    if status == "error":
        return RED
    if status == "running":
        return BLUE
    return DIM


def _to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_finite_number(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


class _SessionPaths:
    def __init__(self, root: Path, session_id: str):
        self.root = root
        self.session_id = session_id
        self.session_dir = root / session_id
        self.meta_path = self.session_dir / "meta.json"
        self.lock_path = self.session_dir / ".lock"
        self.render_path = self.session_dir / ".render.json"
        self.events_path = self.session_dir / "events.jsonl"
        self.states_dir = self.session_dir / "states"

    def state_path(self, slot: int) -> Path:
        return self.states_dir / f"{slot:04d}.json"


class _FileLock:
    def __init__(self, path: Path):
        self.path = path
        self.handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+")
        if fcntl is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)
        elif msvcrt is not None:
            # Windows locks a byte range; ensure byte zero exists first.
            self.handle.seek(0, os.SEEK_END)
            if self.handle.tell() == 0:
                self.handle.write("\0")
                self.handle.flush()
            self.handle.seek(0)
            msvcrt.locking(self.handle.fileno(), msvcrt.LK_LOCK, 1)
        return self.handle

    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None and fcntl is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        elif self.handle is not None and msvcrt is not None:
            self.handle.seek(0)
            msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
        if self.handle is not None:
            self.handle.close()


def _read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return default


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=True))
    os.replace(tmp_path, path)


def _read_last_event_lines(path: Path, limit: int) -> list[str]:
    if limit <= 0 or not path.exists():
        return []

    try:
        raw_lines = path.read_text().splitlines()
    except OSError:
        return []

    messages = []
    for raw in raw_lines[-limit:]:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        stamp = time.strftime("%H:%M:%S", time.localtime(payload.get("ts", time.time())))
        message = str(payload.get("message", "")).strip()
        if message:
            messages.append(f"{stamp}  {message}")
    return messages


def _list_state_files(paths: _SessionPaths) -> list[Path]:
    if not paths.states_dir.exists():
        return []
    return sorted(paths.states_dir.glob("*.json"))


def _terminal_width(default: int = 140) -> int:
    # Leave one column unused to avoid the terminal's automatic line wrapping.
    return max(1, shutil.get_terminal_size((default, 40)).columns - 1)


def _wrap_progress(text: str, width: int, style: str = "") -> str:
    lines = [line for paragraph in text.split("\n")
             for line in (textwrap.wrap(paragraph, width=max(width, 1), subsequent_indent="  " if width > 2 else "",
                                        break_on_hyphens=False) or [""])]
    return "\n".join(f"{style}{line}{RESET}" for line in lines)


def _query_gpu_utils() -> list[tuple[str, int | None]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []

    if result.returncode != 0:
        return []

    values: list[tuple[str, int | None]] = []
    for raw_line in result.stdout.splitlines():
        parts = [part.strip() for part in raw_line.split(",")]
        if not parts or not parts[0]:
            continue
        util: int | None = None
        if len(parts) >= 2:
            try:
                util = int(float(parts[1]))
            except ValueError:
                util = None
        values.append((parts[0], util))
    return values


def _gpu_util_text(values: list[tuple[str, int | None]]) -> str:
    if not values:
        return "gpu util unavailable"
    pieces = []
    for gpu, util in values:
        util_text = "?" if util is None else str(util)
        pieces.append(f"gpu {gpu}: {util_text}%")
    return "  ".join(pieces)


def _refresh_gpu_utils(meta: dict) -> bool:
    now = time.time()
    cached_at = _to_float(meta.get("gpu_util_updated_at"), 0.0)
    cached_values = meta.get("gpu_utils")
    if isinstance(cached_values, list) and now - cached_at < GPU_UTIL_CACHE_SECONDS:
        return False

    values = _query_gpu_utils()
    meta["gpu_utils"] = [{"gpu": gpu, "util": util} for gpu, util in values]
    meta["gpu_util_updated_at"] = now
    return True


def _format_header(meta: dict, states: list[dict], width: int) -> str:
    running = sum(1 for state in states if state.get("status") == "running")
    done = sum(1 for state in states if state.get("status") == "done")
    skipped = sum(1 for state in states if state.get("status") == "skipped")
    raw_gpu_utils = meta.get("gpu_utils")
    gpu_utils: list[tuple[str, int | None]] = []
    if isinstance(raw_gpu_utils, list):
        for item in raw_gpu_utils:
            if not isinstance(item, dict):
                continue
            gpu = str(item.get("gpu", "")).strip()
            if not gpu:
                continue
            util_raw = item.get("util")
            util = int(util_raw) if _is_finite_number(util_raw) else None
            gpu_utils.append((gpu, util))
    timestamp = time.strftime("%H:%M:%S")
    line = f"{_gpu_util_text(gpu_utils)} | running={running} done={done} skipped={skipped} | {timestamp}"
    return _wrap_progress(line, width, BOLD + CYAN)


def _state_fraction(state: dict) -> float:
    current = _to_float(state.get("current"), 0.0)
    total = max(_to_float(state.get("total"), 1.0), 1.0)
    return min(max(current / total, 0.0), 1.0)


def _format_row(state: dict, width: int) -> str:
    slot_value = state.get("slot", 0)
    try:
        slot = int(slot_value) if slot_value is not None else 0
    except (TypeError, ValueError):
        slot = 0
    row_label = str(state.get("row_label") or f"slot {slot:02d}").strip()
    phase = (state.get("phase") or state.get("status") or "idle").lower()
    fraction = _state_fraction(state)
    detail = state.get("detail") or ""
    metric_text = state.get("metric_text") or ""
    bar_width = max(12, min(20, width // 6))
    status_style = _style_for_status(state.get("status", "idle"))

    label = state.get("label") or "waiting"
    line = f"{row_label} | {label} | {phase} [{_bar(fraction, bar_width)}] {fraction * 100:5.1f}%"
    for field in (detail, metric_text):
        if field:
            separator = "  " if len(line.rsplit("\n", 1)[-1]) + 2 + len(field) <= width else "\n"
            line += separator + field

    return _wrap_progress(line, width, status_style)


def _group_status(states: list[dict]) -> str:
    statuses = {str(state.get("status") or "idle").lower() for state in states}
    if "error" in statuses:
        return "error"
    if "running" in statuses:
        return "running"
    if statuses and statuses <= {"done"}:
        return "done"
    if statuses and statuses <= {"skipped"}:
        return "skipped"
    if "done" in statuses or "skipped" in statuses:
        return "running"
    return "idle"


def _group_is_terminal(states: list[dict]) -> bool:
    statuses = {str(state.get("status") or "idle").lower() for state in states}
    return bool(statuses) and statuses <= {"done", "skipped"}


def _group_counts_text(total: int, running: int, done: int, skipped: int) -> str:
    summary = f"{running}/{total} active"
    if done or skipped:
        summary += f"  {done} done"
        if skipped:
            summary += f"  {skipped} skipped"
    return summary


def _terminal_group_status(running: int, done: int, skipped: int) -> str:
    if running > 0:
        return "running"
    if done > 0:
        return "done"
    if skipped > 0:
        return "skipped"
    return "idle"


def _format_group_header_line(
    group_label: str,
    *,
    width: int,
    status: str,
    fraction: float,
    total: int,
    running: int,
    done: int,
    skipped: int,
    group_summary: str = "",
) -> str:
    status_style = _style_for_status(status)
    bar_width = max(12, min(20, width // 6))
    group_width = min(28, max(18, width // 5))
    summary = _group_counts_text(total, running, done, skipped)
    if group_summary:
        line = f"{group_label}  {group_summary}  {summary}"
    else:
        line = (
            f"{group_label:<{group_width}} {status:<7} "
            f"[{_bar(fraction, bar_width)}] {fraction * 100:5.1f}%  {summary}"
        )
    return _wrap_progress(line, width, BOLD + status_style)


def _format_group_header(group_label: str, states: list[dict], width: int, group_summary: str = "") -> str:
    fraction = sum(_state_fraction(state) for state in states) / max(len(states), 1)
    running = sum(1 for state in states if state.get("status") == "running")
    done = sum(1 for state in states if state.get("status") == "done")
    skipped = sum(1 for state in states if state.get("status") == "skipped")
    status = _group_status(states)
    if group_summary and _group_is_terminal(states):
        status = _terminal_group_status(running, done, skipped)
    return _format_group_header_line(
        group_label,
        width=width,
        status=status,
        fraction=fraction,
        total=len(states),
        running=running,
        done=done,
        skipped=skipped,
        group_summary=group_summary,
    )


def _normalize_group_snapshots(raw) -> dict[str, dict]:
    if not isinstance(raw, dict):
        return {}
    normalized: dict[str, dict] = {}
    for key, value in raw.items():
        if not isinstance(value, dict):
            continue
        group_label = str(key or "").strip()
        if not group_label:
            continue
        normalized[group_label] = {
            "summary": str(value.get("summary", "") or ""),
            "status": str(value.get("status", "done") or "done"),
            "fraction": min(max(_to_float(value.get("fraction"), 1.0), 0.0), 1.0),
            "running": max(int(_to_float(value.get("running"), 0.0)), 0),
            "done": max(int(_to_float(value.get("done"), 0.0)), 0),
            "skipped": max(int(_to_float(value.get("skipped"), 0.0)), 0),
            "total": max(int(_to_float(value.get("total"), 0.0)), 0),
        }
    return normalized


def _snapshot_group_state(group_label: str, states: list[dict], summary_text: str = "") -> dict:
    running = sum(1 for state in states if state.get("status") == "running")
    done = sum(1 for state in states if state.get("status") == "done")
    skipped = sum(1 for state in states if state.get("status") == "skipped")
    total = len(states)
    return {
        "summary": str(summary_text or ""),
        "status": _terminal_group_status(running, done, skipped) if _group_is_terminal(states) else _group_status(states),
        "fraction": sum(_state_fraction(state) for state in states) / max(total, 1),
        "running": running,
        "done": done,
        "skipped": skipped,
        "total": total,
    }


def _format_pinned_group_header(group_label: str, snapshot: dict, width: int) -> str:
    return _format_group_header_line(
        group_label,
        width=width,
        status=str(snapshot.get("status", "done") or "done"),
        fraction=min(max(_to_float(snapshot.get("fraction"), 1.0), 0.0), 1.0),
        total=max(int(_to_float(snapshot.get("total"), 0.0)), 0),
        running=max(int(_to_float(snapshot.get("running"), 0.0)), 0),
        done=max(int(_to_float(snapshot.get("done"), 0.0)), 0),
        skipped=max(int(_to_float(snapshot.get("skipped"), 0.0)), 0),
        group_summary=str(snapshot.get("summary", "") or ""),
    )


def _render_rows(
    states: list[dict],
    width: int,
    group_summaries: dict[str, str] | None = None,
    pinned_groups: dict[str, dict] | None = None,
) -> list[str]:
    lines: list[str] = []
    grouped: dict[str, list[dict]] = {}
    group_order: list[str] = []
    ungrouped: list[dict] = []
    summaries = group_summaries or {}
    pinned = pinned_groups or {}
    for state in states:
        group_label = str(state.get("group_label") or "").strip()
        if not group_label:
            ungrouped.append(state)
            continue
        if group_label not in grouped:
            grouped[group_label] = []
            group_order.append(group_label)
        grouped[group_label].append(state)

    for group_label, snapshot in pinned.items():
        if group_label in grouped:
            continue
        lines.append(_format_pinned_group_header(group_label, snapshot, width))

    for group_label in group_order:
        group_states = grouped[group_label]
        group_summary = str(summaries.get(group_label, "") or "")
        lines.append(_format_group_header(group_label, group_states, width, group_summary))
        if group_summary and _group_is_terminal(group_states):
            continue
        lines.extend(_format_row(state, width) for state in group_states)

    lines.extend(_format_row(state, width) for state in ungrouped)
    return lines


def _render_locked(paths: _SessionPaths) -> None:
    if not _supports_live_terminal():
        return

    meta = _read_json(
        paths.meta_path,
        {
            "rows": 0,
            "footer_lines": 0,
            "title": "ResAssetPricing progress",
            "group_summaries": {},
            "pinned_groups": {},
        },
    )
    footer_lines = int(meta.get("footer_lines", 0))
    terminal_width = _terminal_width()
    group_summaries = meta.get("group_summaries", {})
    if not isinstance(group_summaries, dict):
        group_summaries = {}
    pinned_groups = _normalize_group_snapshots(meta.get("pinned_groups", {}))
    if _refresh_gpu_utils(meta):
        _write_json(paths.meta_path, meta)

    states = []
    for state_path in _list_state_files(paths):
        state = _read_json(state_path, None)
        if state is None:
            continue
        states.append(state)
    states.sort(key=lambda state: int(state.get("slot", 0)))

    header = _format_header(meta, states, terminal_width)
    event_lines = _read_last_event_lines(paths.events_path, footer_lines)
    while len(event_lines) < footer_lines:
        event_lines.append("")

    blocks = [header] + _render_rows(states, terminal_width, group_summaries, pinned_groups)
    blocks.extend(_wrap_progress(line, terminal_width) for line in event_lines)
    output_lines = [line for block in blocks for line in block.split("\n")]

    previous_render = _read_json(paths.render_path, {"line_count": 0, "cursor_hidden": False})
    previous_line_count = int(previous_render.get("line_count", 0))

    if not previous_render.get("cursor_hidden", False):
        sys.stdout.write("\033[?25l")

    if previous_line_count > 1:
        sys.stdout.write(f"\033[{previous_line_count - 1}F")
    elif previous_line_count == 1:
        sys.stdout.write("\r")

    rendered_line_count = max(len(output_lines), previous_line_count)
    for index in range(rendered_line_count):
        sys.stdout.write("\033[2K")
        if index < len(output_lines):
            sys.stdout.write(output_lines[index])
        if index != rendered_line_count - 1:
            sys.stdout.write("\n")

    sys.stdout.flush()
    if rendered_line_count > len(output_lines):
        sys.stdout.write(f"\033[{rendered_line_count - len(output_lines)}F")
        sys.stdout.flush()
    _write_json(paths.render_path, {"line_count": len(output_lines), "cursor_hidden": True})


def init_session(
    session_id: str,
    rows: int,
    title: str,
    root: str | None = None,
    footer_lines: int = 0,
    row_templates: list[dict] | None = None,
) -> None:
    if not _supports_live_terminal():
        return

    paths = _SessionPaths(_session_root(root), session_id)
    with _FileLock(paths.lock_path):
        paths.session_dir.mkdir(parents=True, exist_ok=True)
        paths.states_dir.mkdir(parents=True, exist_ok=True)
        previous_meta = _read_json(
            paths.meta_path,
            {
                "rows": 0,
                "footer_lines": 0,
                "title": "ResAssetPricing progress",
                "group_summaries": {},
                "pinned_groups": {},
            },
        )
        previous_summaries = previous_meta.get("group_summaries", {})
        if not isinstance(previous_summaries, dict):
            previous_summaries = {}
        pinned_groups = _normalize_group_snapshots(previous_meta.get("pinned_groups", {}))
        previous_grouped_states: dict[str, list[dict]] = {}
        for state_path in _list_state_files(paths):
            state = _read_json(state_path, None)
            if state is None:
                continue
            group_label = str(state.get("group_label") or "").strip()
            if not group_label:
                continue
            previous_grouped_states.setdefault(group_label, []).append(state)
        for group_label, group_states in previous_grouped_states.items():
            if not _group_is_terminal(group_states):
                continue
            summary_text = str(previous_summaries.get(group_label, "") or pinned_groups.get(group_label, {}).get("summary", ""))
            pinned_groups[group_label] = _snapshot_group_state(group_label, group_states, summary_text)
        for state_path in paths.states_dir.glob("*.json"):
            state_path.unlink(missing_ok=True)
        _write_json(
            paths.meta_path,
            {
                "rows": int(rows),
                "title": title,
                "footer_lines": int(footer_lines),
                "group_summaries": {},
                "pinned_groups": pinned_groups,
                "created_at": time.time(),
            },
        )
        for slot in range(int(rows)):
            template = row_templates[slot] if row_templates and slot < len(row_templates) else {}
            _write_json(
                paths.state_path(slot),
                {
                    "slot": slot,
                    "status": "idle",
                    "group_label": template.get("group_label", ""),
                    "row_label": template.get("row_label", f"slot {slot:02d}"),
                    "label": template.get("label", "waiting"),
                    "phase": template.get("phase", "idle"),
                    "current": 0.0,
                    "total": 1.0,
                    "detail": template.get("detail", "waiting for worker"),
                    "metric_text": template.get("metric_text", ""),
                },
            )
        _render_locked(paths)


def ensure_row_templates(session_id: str, row_templates: list[dict], root: str | None = None) -> None:
    if not _supports_live_terminal() or not session_id or not row_templates:
        return

    paths = _SessionPaths(_session_root(root), session_id)
    with _FileLock(paths.lock_path):
        if not paths.meta_path.exists():
            return
        paths.session_dir.mkdir(parents=True, exist_ok=True)
        paths.states_dir.mkdir(parents=True, exist_ok=True)
        for template in row_templates:
            if "slot" not in template:
                continue
            slot = int(template["slot"])
            state_path = paths.state_path(slot)
            if state_path.exists():
                continue
            _write_json(
                state_path,
                {
                    "slot": slot,
                    "status": template.get("status", "idle"),
                    "group_label": template.get("group_label", ""),
                    "row_label": template.get("row_label", f"slot {slot:02d}"),
                    "label": template.get("label", "waiting"),
                    "phase": template.get("phase", "idle"),
                    "current": 0.0,
                    "total": 1.0,
                    "detail": template.get("detail", "waiting for worker"),
                    "metric_text": template.get("metric_text", ""),
                },
            )
        _render_locked(paths)


def update_row_state(session_id: str, state: dict, root: str | None = None) -> None:
    if not _supports_live_terminal() or not session_id or "slot" not in state:
        return

    paths = _SessionPaths(_session_root(root), session_id)
    with _FileLock(paths.lock_path):
        if not paths.meta_path.exists():
            return
        slot = int(state["slot"])
        existing = _read_json(paths.state_path(slot), {})
        payload = {
            "slot": slot,
            "status": state.get("status", existing.get("status", "running")),
            "group_label": state.get("group_label", existing.get("group_label", "")),
            "row_label": state.get("row_label", existing.get("row_label", f"slot {slot:02d}")),
            "label": state.get("label", existing.get("label", "running")),
            "phase": state.get("phase", existing.get("phase", "running")),
            "current": state.get("current", existing.get("current", 0.0)),
            "total": state.get("total", existing.get("total", 1.0)),
            "detail": state.get("detail", existing.get("detail", "")),
            "metric_text": state.get("metric_text", existing.get("metric_text", "")),
            "updated_at": time.time(),
        }
        _write_json(paths.state_path(slot), payload)
        _render_locked(paths)


def update_group_summary(session_id: str, group_label: str, summary_text: str, root: str | None = None) -> None:
    if not session_id or not group_label:
        return

    paths = _SessionPaths(_session_root(root), session_id)
    with _FileLock(paths.lock_path):
        if not paths.meta_path.exists():
            return
        meta = _read_json(
            paths.meta_path,
            {
                "rows": 0,
                "footer_lines": 0,
                "title": "ResAssetPricing progress",
                "group_summaries": {},
                "pinned_groups": {},
            },
        )
        group_summaries = meta.get("group_summaries", {})
        if not isinstance(group_summaries, dict):
            group_summaries = {}
        pinned_groups = _normalize_group_snapshots(meta.get("pinned_groups", {}))
        text = str(summary_text or "").strip()
        if text:
            group_summaries[str(group_label)] = text
        else:
            group_summaries.pop(str(group_label), None)
        meta["group_summaries"] = group_summaries
        if group_label in pinned_groups:
            pinned_groups[group_label]["summary"] = text
        elif text:
            pinned_groups[group_label] = {
                "summary": text,
                "status": "done",
                "fraction": 1.0,
                "running": 0,
                "done": 0,
                "skipped": 0,
                "total": 0,
            }
        meta["pinned_groups"] = pinned_groups
        _write_json(paths.meta_path, meta)
        _render_locked(paths)


def append_event(session_id: str, message: str, root: str | None = None) -> None:
    if not message:
        return

    paths = _SessionPaths(_session_root(root), session_id)
    with _FileLock(paths.lock_path):
        paths.session_dir.mkdir(parents=True, exist_ok=True)
        with paths.events_path.open("a") as handle:
            handle.write(json.dumps({"ts": time.time(), "message": message}, ensure_ascii=True))
            handle.write("\n")
        _render_locked(paths)


def finalize_session(session_id: str, message: str | None = None, root: str | None = None) -> None:
    if not _supports_live_terminal():
        if message:
            print(message)
        return

    paths = _SessionPaths(_session_root(root), session_id)
    with _FileLock(paths.lock_path):
        if message:
            with paths.events_path.open("a") as handle:
                handle.write(json.dumps({"ts": time.time(), "message": message}, ensure_ascii=True))
                handle.write("\n")
        _render_locked(paths)
        render_state = _read_json(paths.render_path, {"line_count": 0})
        if render_state.get("cursor_hidden", False):
            sys.stdout.write("\033[?25h")
            sys.stdout.flush()
        _write_json(paths.render_path, {"line_count": render_state.get("line_count", 0), "cursor_hidden": False})
        sys.stdout.write("\n")
        sys.stdout.flush()


class JobProgress:
    def __init__(self, session_id: str | None, slot: int | None, label: str | None = None):
        self.session_id = session_id
        self.slot = slot
        self.label = label or os.environ.get("ResAssetPricing_PROGRESS_LABEL") or "job"
        self.group_label = os.environ.get("ResAssetPricing_PROGRESS_GROUP_LABEL")
        self.row_label = os.environ.get("ResAssetPricing_PROGRESS_ROW_LABEL")
        self.root = _session_root()
        self.live_terminal = _supports_live_terminal()
        self.multi_process = self.live_terminal and self.session_id is not None and self.slot is not None
        self.standalone = self.live_terminal and not self.multi_process
        self.last_total = 1.0
        self.last_completed = 0.0
        self.rendered_lines = 0

    @classmethod
    def from_env(cls, fallback_label: str | None = None):
        session_id = os.environ.get("ResAssetPricing_PROGRESS_SESSION")
        slot_raw = os.environ.get("ResAssetPricing_PROGRESS_SLOT")
        slot = int(slot_raw) if slot_raw is not None else None
        return cls(session_id=session_id, slot=slot, label=os.environ.get("ResAssetPricing_PROGRESS_LABEL") or fallback_label)

    def update(
        self,
        *,
        phase: str,
        completed: float,
        total: float,
        detail: str = "",
        metric_text: str = "",
        status: str = "running",
        label: str | None = None,
    ) -> None:
        total = max(float(total), 1.0)
        completed = min(max(float(completed), 0.0), total)
        self.last_total = total
        self.last_completed = completed

        payload = {
            "slot": self.slot,
            "status": status,
            "group_label": self.group_label or "",
            "row_label": self.row_label or (f"slot {self.slot:02d}" if self.slot is not None else "job"),
            "label": label or self.label,
            "phase": phase,
            "current": completed,
            "total": total,
            "detail": detail,
            "metric_text": metric_text,
            "updated_at": time.time(),
        }

        if self.multi_process:
            paths = _SessionPaths(self.root, self.session_id)
            with _FileLock(paths.lock_path):
                _write_json(paths.state_path(self.slot), payload)
                _render_locked(paths)
            return

        if self.standalone:
            width = _terminal_width()
            lines = _format_row(payload, width).split("\n")
            if self.rendered_lines > 1:
                sys.stdout.write(f"\033[{self.rendered_lines - 1}F")
            else:
                sys.stdout.write("\r")
            count = max(len(lines), self.rendered_lines)
            sys.stdout.write("\n".join("\033[2K" + (lines[i] if i < len(lines) else "") for i in range(count)))
            if count > len(lines):
                sys.stdout.write(f"\033[{count - len(lines)}F")
            self.rendered_lines = len(lines)
            sys.stdout.flush()
            if status in {"done", "skipped", "error"}:
                sys.stdout.write("\n")
                sys.stdout.flush()
                self.rendered_lines = 0

    def mark_done(self, detail: str = "", metric_text: str = "") -> None:
        self.update(
            phase="done",
            completed=self.last_total,
            total=self.last_total,
            detail=detail,
            metric_text=metric_text,
            status="done",
        )

    def mark_skipped(self, *, completed: float, total: float, detail: str) -> None:
        self.update(
            phase="skip",
            completed=completed,
            total=total,
            detail=detail,
            metric_text="",
            status="skipped",
        )

    def mark_error(self, detail: str) -> None:
        self.update(
            phase="error",
            completed=self.last_completed,
            total=self.last_total,
            detail=detail,
            metric_text="",
            status="error",
        )


class SummaryPrinter:
    def __init__(self, session_id: str | None = None, root: str | None = None):
        self.session_id = session_id or os.environ.get("ResAssetPricing_PROGRESS_SESSION")
        self.root = root or os.environ.get("ResAssetPricing_PROGRESS_ROOT")

    @classmethod
    def from_env(cls):
        return cls()

    def emit(self, message: str) -> None:
        if not message:
            return
        if self.session_id and _supports_live_terminal():
            append_event(self.session_id, message, self.root)
        else:
            print(message)

    def emit_econ_result(
        self,
        *,
        asset_prefix: str,
        group: str,
        sr_map: dict[str, float],
    ) -> None:
        subject = asset_prefix

        grouped_live = self.session_id and os.environ.get("ResAssetPricing_PROGRESS_GROUPED") == "1"
        if grouped_live:
            if group == "ALL":
                preferred_scheme = None
                if _is_finite_number(sr_map.get("vw")):
                    preferred_scheme = "vw"
                elif _is_finite_number(sr_map.get("ew")):
                    preferred_scheme = "ew"
                if preferred_scheme is None:
                    summary_text = "ALL SR unavailable"
                else:
                    summary_text = f"ALL SR {preferred_scheme} {float(sr_map[preferred_scheme]):+0.3f}"
                update_group_summary(self.session_id, asset_prefix, summary_text, self.root)
            return

        pieces = []
        for scheme in ("ew", "vw"):
            value = sr_map.get(scheme)
            if _is_finite_number(value):
                pieces.append(f"{scheme} {float(value):+0.3f}")

        if pieces:
            message = f"[econ] {_truncate(subject, 34):<34} {_truncate(group, 12):<12} SR  {'  '.join(pieces)}"
        else:
            message = f"[econ] {_truncate(subject, 34):<34} {_truncate(group, 12):<12} SR  unavailable"
        self.emit(message)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("--session", required=True)
    init_parser.add_argument("--rows", required=True, type=int)
    init_parser.add_argument("--title", required=True)
    init_parser.add_argument("--root", default=None)
    init_parser.add_argument("--footer-lines", default=0, type=int)

    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--session", required=True)
    finalize_parser.add_argument("--root", default=None)
    finalize_parser.add_argument("--message", default="")

    event_parser = subparsers.add_parser("event")
    event_parser.add_argument("--session", required=True)
    event_parser.add_argument("--root", default=None)
    event_parser.add_argument("--message", required=True)

    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "init":
        init_session(args.session, args.rows, args.title, args.root, args.footer_lines)
        return 0
    if args.command == "finalize":
        finalize_session(args.session, args.message, args.root)
        return 0
    if args.command == "event":
        append_event(args.session, args.message, args.root)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
