"""Prepare each refinement exhibit from the shared pair-count analysis."""
from utils.paper_source.pairwise_cache import load_group_summary, load_depth_summary
import numpy as np
import pandas as pd
from utils.paper_source.paths import SOURCE_DIR


def analyze_forecast_refinement_table() -> dict:
    return {"table": load_group_summary()}


def analyze_forecast_rank_preservation_and_refinement_by_depth() -> dict:
    return {"depth_summary": load_depth_summary()}


def analyze_theme_pairwise_losses_table() -> dict:
    """Package P/C/V losses with the same theme removed from both forecasts."""
    frame = pd.read_csv(SOURCE_DIR / "theme_pairwise_loss_summary.csv")
    if "AnchorCondition" not in frame or not frame.AnchorCondition.eq("same_theme_zeroed").all():
        raise ValueError("Table 5 requires both shallow and deep theme removal; run analysis.theme_refinement.")
    keys = ["Model", "Theme", "Metric"]
    if len(frame) != 156 or frame.duplicated(keys).any():
        raise ValueError("Expected 13 themes by four models by three measures.")
    if set(frame.Model) != {"ResNet", "NN", "ResNet+", "NN+"} or set(frame.Metric) != {"P", "C", "V"}:
        raise ValueError("Unexpected model or measure in theme-loss source.")
    if frame.Theme.nunique() != 13 or not frame.groupby(["Model", "Metric"]).size().eq(13).all():
        raise ValueError("Incomplete theme grid.")
    if not np.isfinite(frame[["display_mean", "holm_p"]].to_numpy()).all() or not frame.holm_p.between(0, 1).all():
        raise ValueError("Invalid estimates or adjusted p-values.")
    return {"table": frame}
