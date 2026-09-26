"""Device selection shared by the shell launcher, TUI and training entry point."""
from __future__ import annotations

import argparse
import os
import platform
import sys


def _torch():
    try:
        import torch
    except ImportError as exc:
        if sys.platform == "darwin":
            wheel = "uv pip install -r docker/requirements/torch-macos.txt"
        else:
            wheel = "uv pip install torch==2.7.1"
            wheel += " --index-url https://download.pytorch.org/whl/cpu"
        raise RuntimeError(
            f"PyTorch is unavailable in {sys.executable}. Activate your uv environment, "
            f"install the project dependencies, then run `{wheel}`. "
            "For NVIDIA GPUs on Linux/Windows, use the cu128 wheel index instead of cpu."
        ) from exc
    return torch


def available_devices() -> list[str]:
    torch = _torch()
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        visible = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
        if count > 0:
            if len(visible) >= count and "-1" not in visible[:count]:
                return visible[:count]
            return [str(index) for index in range(count)]
    if torch.backends.mps.is_available():
        return ["mps"]
    return ["cpu"]


def worker_device(device: str) -> str:
    """CUDA workers see one masked GPU, so its in-process index is always zero."""
    return device if device in {"cpu", "mps"} else "cuda:0"


def worker_environment(device: str) -> dict[str, str]:
    return {"CUDA_VISIBLE_DEVICES": "" if device in {"cpu", "mps"} else device}


def default_device() -> str:
    return worker_device(available_devices()[0])


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return default_device()
    torch = _torch()
    try:
        device = torch.device(requested)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"Invalid device: {requested}. Use auto, cpu, mps or cuda:N.") from exc
    if device.type == "cpu":
        return "cpu"
    if device.type == "mps":
        if torch.backends.mps.is_available():
            return "mps"
        raise ValueError("Apple MPS is unavailable. Use --device cpu or install an MPS-enabled PyTorch on a supported Mac.")
    if device.type == "cuda":
        index = device.index or 0
        if torch.cuda.is_available() and index < torch.cuda.device_count():
            return f"cuda:{index}"
        raise ValueError(f"CUDA device {index} is unavailable. Check the driver/visible devices or use --device cpu.")
    raise ValueError(f"Unsupported device: {requested}. Use auto, cpu, mps or cuda:N.")


def select_devices(requested: str = "", count: int = 1) -> list[str]:
    if count < 1:
        raise ValueError("Device count must be positive.")
    if requested.strip() in {"cpu", "mps"}:
        return [resolve_device(requested.strip())]
    available = available_devices()
    if requested:
        selected = [token.strip() for token in requested.split(",") if token.strip()]
        if not selected or len(set(selected)) != len(selected) or any(token not in available for token in selected):
            raise ValueError(f"Requested devices {requested!r} are unavailable. Available: {', '.join(available)}.")
        return selected
    if available[0] in {"cpu", "mps"}:
        return available
    if count > len(available):
        raise ValueError(f"Requested {count} CUDA GPUs, but only {len(available)} are available.")
    return available[:count]


def default_slots(device: str) -> int:
    return 1 if device == "cpu" else 10 if device == "mps" else 20


def device_slots(device: str, requested: int) -> int:
    # Apply the platform limit when loading plans saved with a different cap.
    return default_slots(device) if device in {"cpu", "mps"} else max(int(requested), 1)


def device_message(devices: list[str]) -> str:
    system = "macOS" if sys.platform == "darwin" else platform.system()
    if system == "Linux" and "microsoft" in platform.release().lower():
        system = "Windows / WSL2"
    if devices == ["mps"]:
        return f"{system}: using Apple GPU (MPS); up to {default_slots('mps')} concurrent tasks."
    if devices == ["cpu"]:
        if available_devices() != ["cpu"]:
            return f"{system}: CPU selected; one task at a time."
        if sys.platform == "darwin":
            return "macOS: Apple GPU (MPS) is unavailable. Using CPU; one task at a time."
        return f"{system}: no usable GPU detected. Using CPU; one task at a time."
    return f"{system}: using NVIDIA CUDA device(s) {', '.join(devices)}."


def empty_device_cache(device: str) -> None:
    torch = _torch()
    kind = torch.device(device).type
    if kind == "cuda":
        torch.cuda.empty_cache()
    elif kind == "mps":
        torch.mps.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--select", default="")
    parser.add_argument("--count", type=int, default=1)
    args = parser.parse_args()
    try:
        devices = select_devices(args.select, args.count)
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(device_message(devices), file=sys.stderr)
    print(" ".join(devices))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
