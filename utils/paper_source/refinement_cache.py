"""Load refinement exhibit caches or summary CSVs from a specified directory."""
from pathlib import Path
import pandas as pd
from utils.paper_source.artifact_cache import load_render_cache as load_cache
from utils.paper_source.pairwise_cache import _read_summary


def load_group_summary(cache_dir: Path | None = None) -> pd.DataFrame:
    if cache_dir is not None:
        return _read_summary("group_summary.csv", cache_dir)
    return load_cache("ForecastRefinement")["table"]


def load_depth_summary(cache_dir: Path | None = None) -> pd.DataFrame:
    if cache_dir is not None:
        return _read_summary("depth_summary.csv", cache_dir)
    return load_cache("ForecastRankPreservationAndRefinementByDepth")["depth_summary"]
