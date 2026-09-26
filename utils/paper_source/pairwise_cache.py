"""Read refinement summary CSVs without running analysis."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.paper_source.paths import SOURCE_DIR


DEFAULT_CACHE_DIR = SOURCE_DIR / "pairwise_refinement"


def _read_summary(name: str, cache_dir: Path | None = None) -> pd.DataFrame:
    directory = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    return pd.read_csv(directory / name)


def load_group_summary(cache_dir: Path | None = None) -> pd.DataFrame:
    return _read_summary("group_summary.csv", cache_dir)


def load_depth_summary(cache_dir: Path | None = None) -> pd.DataFrame:
    return _read_summary("depth_summary.csv", cache_dir)
