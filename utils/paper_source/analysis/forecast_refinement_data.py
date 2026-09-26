"""Pure nested-sort reference retained for the historical pairwise audit.

Current manuscript preservation and refinement use analysis.forecast_refinement.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rank_score(values: np.ndarray) -> np.ndarray:
    ranks = pd.Series(values, copy=False).rank(method="average").to_numpy(dtype=float)
    standard_deviation = float(ranks.std(ddof=0))
    if not np.isfinite(standard_deviation) or standard_deviation <= 0.0:
        raise ValueError("Forecast ranks must have positive cross-sectional dispersion.")
    return (ranks - float(ranks.mean())) / standard_deviation


def _deterministic_groups(signal: np.ndarray, groups: int) -> np.ndarray:
    if len(signal) < groups:
        raise ValueError("The cross section is too small for the requested portfolio groups.")
    order = np.argsort(signal, kind="mergesort")
    labels = np.empty(len(signal), dtype=np.int16)
    labels[order] = np.minimum(
        np.arange(len(signal), dtype=np.int64) * int(groups) // len(signal),
        int(groups) - 1,
    )
    return labels


def _value_weighted_return(
    realized: np.ndarray,
    log_market_equity: np.ndarray,
) -> float:
    shifted = log_market_equity - float(np.max(log_market_equity))
    weights = np.exp(shifted)
    denominator = float(weights.sum())
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("Value weights must have a positive finite sum.")
    return float(np.dot(weights, realized) / denominator)


def _conditional_correction_return(
    anchor_score: np.ndarray,
    correction: np.ndarray,
    realized: np.ndarray,
    log_market_equity: np.ndarray,
) -> float:
    anchor_groups = _deterministic_groups(anchor_score, 10)
    spreads: list[float] = []
    for anchor_group in range(10):
        inside = anchor_groups == anchor_group
        if int(inside.sum()) < 30:
            raise ValueError("Each shallow forecast decile must contain at least 30 stocks.")
        correction_groups = _deterministic_groups(correction[inside], 3)
        low = correction_groups == 0
        high = correction_groups == 2
        spreads.append(
            _value_weighted_return(
                realized[inside][high],
                log_market_equity[inside][high],
            )
            - _value_weighted_return(
                realized[inside][low],
                log_market_equity[inside][low],
            )
        )
    return float(np.mean(spreads))


def _model_statistics(
    anchor: np.ndarray,
    target: np.ndarray,
    realized: np.ndarray,
    log_market_equity: np.ndarray,
) -> tuple[float, float, bool]:
    anchor_score = _rank_score(anchor)
    if float(np.nanmax(target) - np.nanmin(target)) <= 0.0:
        return 0.0, 0.0, True
    target_score = _rank_score(target)
    rank_correlation = float(np.mean(anchor_score * target_score))
    correction = target_score - rank_correlation * anchor_score
    correction_standard_deviation = float(correction.std(ddof=0))
    if not np.isfinite(correction_standard_deviation) or correction_standard_deviation <= 0.0:
        raise ValueError("The noncollapsed forecast correction has no cross-sectional dispersion.")
    correction_return = _conditional_correction_return(
        anchor_score,
        correction,
        realized,
        log_market_equity,
    )
    return rank_correlation, correction_return, False
