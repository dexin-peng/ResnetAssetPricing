from __future__ import annotations

import importlib
import inspect
import re
from pathlib import Path

from config.seeded_static import MAX_HIDDEN_LAYER_COUNT, SEED_WIDTH_TEMPLATES


_CONFIG_DIR = Path(__file__).resolve().parent
_EXPECTED_COMBINATIONS_PER_PACKAGE = sum(MAX_HIDDEN_LAYER_COUNT + 1 - seed for seed in SEED_WIDTH_TEMPLATES)
_PACKAGE_LAYOUTS = (
    (
        "resnetp",
        re.compile(r"^resnetps([1-4])d([1-9][0-9]*)\.py$"),
        lambda seed, depth: f"ResNetpS{seed}D{depth}Config",
    ),
    (
        "nnp",
        re.compile(r"^nnps([1-4])d([1-9][0-9]*)\.py$"),
        lambda seed, depth: f"NNpS{seed}D{depth}Config",
    ),
    (
        "resnet",
        re.compile(r"^resnets([1-4])d([1-9][0-9]*)\.py$"),
        lambda seed, depth: f"ResNetS{seed}D{depth}Config",
    ),
    (
        "nn",
        re.compile(r"^nns([1-4])d([1-9][0-9]*)\.py$"),
        lambda seed, depth: f"NNS{seed}D{depth}Config",
    ),
)


def _load_config_class(module_name: str):
    module = importlib.import_module(module_name)
    config_classes = [
        obj
        for _, obj in inspect.getmembers(module, inspect.isclass)
        if obj.__module__ == module.__name__ and obj.__name__.endswith("Config")
    ]
    if len(config_classes) != 1:
        raise RuntimeError(f"Expected exactly one config class in {module_name}, got {len(config_classes)}")
    return config_classes[0]


def _iter_seeded_modules(package_dir: str, stem_pattern: re.Pattern[str]):
    package_path = _CONFIG_DIR / package_dir
    entries = []
    for module_path in sorted(package_path.glob("*.py")):
        match = stem_pattern.fullmatch(module_path.name)
        if not match:
            continue

        seed = int(match.group(1))
        depth = int(match.group(2))
        if depth < seed:
            raise RuntimeError(f"Invalid seeded config module {module_path.name}: expected depth >= seed")
        if depth > MAX_HIDDEN_LAYER_COUNT:
            raise RuntimeError(
                f"Invalid seeded config module {module_path.name}: "
                f"expected depth <= {MAX_HIDDEN_LAYER_COUNT}"
            )
        entries.append((seed, depth, module_path.stem))

    entries.sort(key=lambda item: (item[0], item[1], item[2]))
    if len(entries) != _EXPECTED_COMBINATIONS_PER_PACKAGE:
        raise RuntimeError(
            f"Expected {_EXPECTED_COMBINATIONS_PER_PACKAGE} seeded configs in {package_dir}, got {len(entries)}"
        )
    return entries


def _register_package(package_dir: str, stem_pattern: re.Pattern[str], expected_class_name):
    package_name = f"config.{package_dir.replace('/', '.')}"
    registered = {}

    for seed, depth, module_stem in _iter_seeded_modules(package_dir, stem_pattern):
        config_cls = _load_config_class(f"{package_name}.{module_stem}")
        expected_name = expected_class_name(seed, depth)
        if config_cls.__name__ != expected_name:
            raise RuntimeError(
                f"Unexpected config class name in {package_name}.{module_stem}: "
                f"expected {expected_name}, got {config_cls.__name__}"
            )
        if config_cls.__name__ in registered:
            raise RuntimeError(f"Duplicate config class {config_cls.__name__} in {package_dir}")
        registered[config_cls.__name__] = config_cls

    return registered


REGISTERED_MODEL_CONFIGS = {}
for package_dir, stem_pattern, expected_class_name in _PACKAGE_LAYOUTS:
    package_configs = _register_package(package_dir, stem_pattern, expected_class_name)
    overlapping_names = REGISTERED_MODEL_CONFIGS.keys() & package_configs.keys()
    if overlapping_names:
        overlapping_names_text = ", ".join(sorted(overlapping_names))
        raise RuntimeError(f"Duplicate config classes across packages: {overlapping_names_text}")
    REGISTERED_MODEL_CONFIGS.update(package_configs)

PUBLIC_MODEL_CONFIGS = dict(REGISTERED_MODEL_CONFIGS)
PUBLIC_MODEL_CONFIG_CHOICES = tuple(PUBLIC_MODEL_CONFIGS.keys())


def resolve_config_cls(config_name: str):
    if config_name in PUBLIC_MODEL_CONFIGS:
        return PUBLIC_MODEL_CONFIGS[config_name]
    raise KeyError(config_name)
