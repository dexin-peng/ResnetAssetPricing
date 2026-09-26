"""Read-only environment probe; kept compatible with system Python 3.9.

Exit codes: 10 = Python needs replacing, 20 = packages need installing,
21 = installed packages fail an import/runtime smoke check.
"""
import argparse
import importlib
from importlib import metadata
from pathlib import Path
import platform
import struct
import sys


def check_python(project_venv=None):
    print(f"Python {platform.python_version()}: {sys.executable}", flush=True)
    # NumPy 2.5 on macOS needs Python >= 3.12; SciPy 1.16 elsewhere needs >= 3.11.
    min_minor = 12 if sys.platform == "darwin" else 11
    if not (3, min_minor) <= sys.version_info[:2] <= (3, 13):
        print(f"The pinned wheels support Python 3.{min_minor}-3.13; setup uses 3.13.5.")
        return False
    if struct.calcsize("P") != 8:
        print("The pinned dependencies require 64-bit Python.")
        return False
    if project_venv is not None and Path(sys.prefix).resolve() != project_venv.resolve():
        print("The project interpreter points outside .venv; create an isolated project environment.")
        return False
    if sys.platform == "darwin" and platform.machine() != "arm64":
        print("This Python runs as Intel/x86_64; the native PyTorch wheel needs arm64.")
        return False
    try:
        import ssl
        if sys.platform != "win32":
            import curses
    except ImportError as exc:
        print(f"Python is missing terminal/TLS support: {exc}")
        return False
    return True


def check_packages(root, profile):
    try:
        from packaging.requirements import Requirement
    except ImportError:
        print("Missing dependency metadata support: packaging")
        return 20
    problems = []
    torch_file = {"macos": "torch-macos.txt", "cpu": "torch-cpu.txt", "cuda": "torch.txt"}[profile]
    files = ["common.txt", "data.txt", torch_file]
    if sys.platform == "win32":
        files.append("windows.txt")
    for filename in files:
        for line in (root / "docker" / "requirements" / filename).read_text().splitlines():
            requirement = line.split("#", 1)[0].strip()
            if not requirement:
                continue
            parsed = Requirement(requirement)
            if parsed.marker and not parsed.marker.evaluate():
                continue
            name = parsed.name
            expected = next(iter(parsed.specifier)).version
            try:
                actual = metadata.version(name)
            except metadata.PackageNotFoundError:
                problems.append(f"Missing: {requirement}")
            else:
                # A CUDA wheel can also run on CPU (e.g. on an HPC login node).
                matches = actual == expected
                if name == "torch" and profile == "cpu":
                    matches = actual.split("+")[0] == expected.split("+")[0]
                if not matches:
                    problems.append(f"Version differs: {name} {actual} (expected {expected})")
    if problems:
        print("\n".join(problems))
        return 20
    # Metadata alone does not detect broken wheels, binary incompatibility, or
    # a missing transitive dependency. Exercise the numerical stack, too.
    try:
        for module in ("curses", "numpy", "pandas", "scipy.linalg", "statsmodels.api",
                       "matplotlib", "ml_collections", "openpyxl", "tqdm",
                       "torch", "torch.utils.tensorboard"):
            importlib.import_module(module)
        import numpy as np
        import torch
        torch.from_numpy(np.zeros(1, dtype=np.float32)).numpy()
    except Exception as exc:
        print(f"Dependency check failed: {type(exc).__name__}: {exc}")
        return 21
    if profile == "cuda" and not torch.cuda.is_available():
        print("CUDA cannot be used by PyTorch. Check the NVIDIA driver / container GPU access; CPU will be used.")
    from utils.runtime_device import available_devices, device_message
    print(device_message(available_devices()))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--python-only", action="store_true")
    parser.add_argument("--project-venv", action="store_true")
    parser.add_argument("--profile", choices=("macos", "cpu", "cuda"), required=True)
    args = parser.parse_args()
    if not check_python(args.project_root / ".venv" if args.project_venv else None):
        return 10
    if args.python_only:
        return 0
    sys.path.insert(0, str(args.project_root))
    return check_packages(args.project_root, args.profile)


if __name__ == "__main__":
    raise SystemExit(main())
