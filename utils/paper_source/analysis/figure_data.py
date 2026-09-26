"""Numerical figure inputs. Plotting code consumes only their completed caches."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .data import (
    CONSTANT_PREDICT_FALLBACK_COLUMN,
    completed_results,
    filter_noncollapsed_rows,
    load_brief_results,
    load_decile_returns,
)
from utils.paper_source.figure_spec import (
    PARAMETER_SCALE_EXTERNAL_RECORDS,
    VW_SR,
)


def _filter_to_cells(frame: pd.DataFrame, cells: list[tuple[int, int]]) -> pd.DataFrame:
    if frame.empty or not cells:
        return frame.iloc[0:0].copy()
    out = frame.copy()
    out["seed"] = pd.to_numeric(out["seed"], errors="coerce")
    out["depth"] = pd.to_numeric(out["depth"], errors="coerce")
    out = out.dropna(subset=["seed", "depth"])
    out["seed"] = out["seed"].astype(int)
    out["depth"] = out["depth"].astype(int)
    cell_index = pd.MultiIndex.from_tuples(cells, names=["seed", "depth"])
    row_index = pd.MultiIndex.from_frame(out[["seed", "depth"]])
    return out[row_index.isin(cell_index)].copy()


def _average_decile_risk_statistics(
    decile: pd.DataFrame,
    *,
    family: str,
    cells: list[tuple[int, int]],
) -> pd.DataFrame:
    labels = [str(i) for i in range(1, 11)]
    part = decile[
        (decile["family"] == family)
        & (decile["group"] == "ALL")
        & (decile["weighting"] == "VW")
        & (decile["decile"].astype(str).isin(labels))
    ].copy()
    part = _filter_to_cells(part, cells)
    if part.empty:
        return pd.DataFrame(columns=["Volatility", "SR"])
    part["decile_num"] = pd.to_numeric(part["decile"], errors="coerce")
    part["return"] = pd.to_numeric(part["return"], errors="coerce")
    cell_statistics = (
        part.dropna(subset=["decile_num", "return"])
        .groupby(["seed", "depth", "decile_num"])["return"]
        .agg(["mean", "std"])
        .reset_index()
    )
    cell_statistics["Volatility"] = cell_statistics["std"] * 100.0
    cell_statistics["SR"] = np.where(
        cell_statistics["std"] > 0.0,
        cell_statistics["mean"] / cell_statistics["std"] * np.sqrt(12.0),
        np.nan,
    )
    return (
        cell_statistics.groupby("decile_num")[["Volatility", "SR"]]
        .mean()
        .reindex(range(1, 11))
    )


def _decile_mean_panel(decile: pd.DataFrame) -> pd.DataFrame:
    labels = [str(i) for i in range(1, 11)]
    part = decile[
        (decile["group"] == "ALL")
        & (decile["weighting"] == "VW")
        & (decile["decile"].astype(str).isin(labels))
    ][["family", "seed", "depth", "decile", "return"]].copy()
    if part.empty:
        return pd.DataFrame(columns=["family", "seed", "depth", "decile_num", "return"])
    part["seed"] = pd.to_numeric(part["seed"], errors="coerce")
    part["depth"] = pd.to_numeric(part["depth"], errors="coerce")
    part["decile_num"] = pd.to_numeric(part["decile"], errors="coerce")
    part["return"] = pd.to_numeric(part["return"], errors="coerce")
    part = part.dropna(subset=["seed", "depth", "decile_num", "return"])
    part["seed"] = part["seed"].astype(int)
    part["depth"] = part["depth"].astype(int)
    part["decile_num"] = part["decile_num"].astype(int)
    return (
        part.groupby(["family", "seed", "depth", "decile_num"], sort=False)["return"]
        .mean()
        .mul(100.0)
        .reset_index()
    )


def _completed_paired_cells_from_decile_means(
    decile_means: pd.DataFrame,
    *,
    left_family: str,
    right_family: str,
) -> list[tuple[int, int]]:
    if decile_means.empty:
        return []
    coverage = (
        decile_means[decile_means["family"].isin([left_family, right_family])]
        .drop_duplicates(["family", "seed", "depth", "decile_num"])
        .groupby(["family", "seed", "depth"])["decile_num"]
        .nunique()
        .reset_index(name="n_deciles")
    )
    coverage = coverage[coverage["n_deciles"] >= 10]
    left_cells = set(
        coverage[coverage["family"] == left_family][["seed", "depth"]]
        .itertuples(index=False, name=None)
    )
    right_cells = set(
        coverage[coverage["family"] == right_family][["seed", "depth"]]
        .itertuples(index=False, name=None)
    )
    return sorted((int(seed), int(depth)) for seed, depth in left_cells & right_cells)


def _average_decile_returns_from_means(
    decile_means: pd.DataFrame,
    *,
    family: str,
    cells: list[tuple[int, int]],
) -> pd.Series:
    if decile_means.empty or not cells:
        return pd.Series(dtype=float)
    part = decile_means[decile_means["family"] == family].copy()
    cell_index = pd.MultiIndex.from_tuples(cells, names=["seed", "depth"])
    row_index = pd.MultiIndex.from_frame(part[["seed", "depth"]])
    part = part[row_index.isin(cell_index)]
    if part.empty:
        return pd.Series(dtype=float)
    return (
        part.groupby("decile_num")["return"]
        .mean()
        .reindex(range(1, 11))
    )


def _parameter_scale_records() -> pd.DataFrame:
    brief = load_brief_results(include_fallback=True)
    records: list[dict[str, object]] = []
    own_groups = [
        ("NN/ResNet grid", ("NN", "ResNet")),
        ("ResNet+/NN+ grid", ("ResNet+", "NN+")),
    ]
    for label, families in own_groups:
        values = pd.to_numeric(
            brief.loc[brief["family"].isin(families), "Trainable Parameters"],
            errors="coerce",
        ).dropna()
        if values.empty:
            continue
        records.append(
            {
                "label": label,
                "domain": "Asset Pricing",
                "low": float(values.min()),
                "high": float(values.max()),
                "year": "",
            }
        )
    records.extend(PARAMETER_SCALE_EXTERNAL_RECORDS)
    out = pd.DataFrame(records)
    if not out.empty:
        out["low"] = pd.to_numeric(out["low"], errors="coerce")
        out["high"] = pd.to_numeric(out["high"], errors="coerce")
        out = out.dropna(subset=["low", "high"])
        out = out.sort_values(["high", "low"]).reset_index(drop=True)
    return out


def _delta_metric_matrix(
    left_family: str,
    right_family: str,
    metric: str,
) -> tuple[pd.DataFrame, float]:
    raw = completed_results((left_family, right_family), include_fallback=True).copy()
    df = filter_noncollapsed_rows(raw)
    keep = ["family", "seed", "depth", metric]
    wide = df[keep].pivot_table(index=["seed", "depth"], columns="family", values=metric, aggfunc="first")
    if left_family not in wide or right_family not in wide:
        return pd.DataFrame(), np.nan
    delta = wide[left_family] - wide[right_family]
    matrix = delta.unstack("depth").sort_index()
    values = np.abs(matrix.to_numpy(dtype=float)) if not matrix.empty else np.array([])
    finite_values = values[np.isfinite(values)]
    max_abs = float(finite_values.max()) if finite_values.size else np.nan
    return matrix, max_abs


def _constant_fallback_paired_cells(
    left_family: str,
    right_family: str,
    metric: str,
) -> list[tuple[int, int]]:
    raw = completed_results((left_family, right_family), include_fallback=True).copy()
    if raw.empty:
        return []
    raw = raw.dropna(subset=[metric, "seed", "depth"])
    if raw.empty:
        return []
    if CONSTANT_PREDICT_FALLBACK_COLUMN not in raw.columns:
        return []
    raw["constant_fallback"] = pd.to_numeric(
        raw[CONSTANT_PREDICT_FALLBACK_COLUMN],
        errors="coerce",
    ).fillna(0.0) > 0.0
    metric_wide = raw.pivot_table(
        index=["seed", "depth"],
        columns="family",
        values=metric,
        aggfunc="first",
    )
    fallback_wide = raw.pivot_table(
        index=["seed", "depth"],
        columns="family",
        values="constant_fallback",
        aggfunc="max",
    )
    if left_family not in metric_wide or right_family not in metric_wide:
        return []
    if left_family not in fallback_wide or right_family not in fallback_wide:
        return []
    cells = metric_wide.dropna(subset=[left_family, right_family]).index
    out: list[tuple[int, int]] = []
    for seed, depth in cells:
        if bool(fallback_wide.loc[(seed, depth), left_family]) or bool(
            fallback_wide.loc[(seed, depth), right_family]
        ):
            out.append((int(seed), int(depth)))
    return sorted(out)


def _delta_heatmap_payload(
    *,
    left_family: str,
    right_family: str,
    metric: str,
) -> tuple[pd.DataFrame, list[int], list[int], float]:
    matrix, max_abs = _delta_metric_matrix(left_family, right_family, metric)
    if matrix.empty:
        return matrix, [], [], np.nan
    depths = list(range(int(matrix.columns.min()), int(matrix.columns.max()) + 1))
    seeds = list(range(int(matrix.index.min()), int(matrix.index.max()) + 1))
    matrix = matrix.reindex(index=seeds, columns=depths)
    return matrix, depths, seeds, max_abs


def analyze_rolling_train_test_split() -> dict | None:
    forecast_years = np.arange(1987, 2024)
    train_x: list[int] = []
    train_y: list[int] = []
    test_x: list[int] = []
    test_y: list[int] = []
    for year in forecast_years:
        train_years = np.arange(1963, year)
        train_x.extend(train_years.tolist())
        train_y.extend([int(year)] * len(train_years))
        test_x.append(int(year))
        test_y.append(int(year))

    return {'train_x': train_x, 'train_y': train_y, 'test_x': test_x, 'test_y': test_y}


def analyze_parameter_scale_gap() -> dict:
    return {"records": _parameter_scale_records()}


def analyze_decile_monotonicity_selected() -> dict:
    decile = load_decile_returns(include_fallback=True)
    means = _decile_mean_panel(decile)
    values = {}
    for left, right in (("ResNet", "NN"), ("ResNet+", "NN+")):
        cells = _completed_paired_cells_from_decile_means(means, left_family=left, right_family=right)
        for family in (left, right):
            risk = _average_decile_risk_statistics(decile, family=family, cells=cells)
            values[family] = {"Mean": _average_decile_returns_from_means(means, family=family, cells=cells),
                              "Volatility": risk["Volatility"], "SR": risk["SR"]}
    return {"values": values}


def _heatmap_data(left: str, right: str) -> dict:
    metric = VW_SR
    matrix, depths, seeds, max_abs = _delta_heatmap_payload(left_family=left, right_family=right, metric=metric)
    return {"matrix": matrix, "depths": depths, "seeds": seeds, "max_abs": max_abs,
            "fallback_cells": _constant_fallback_paired_cells(left, right, metric)}


def analyze_specification_delta_sr_heatmaps() -> dict:
    return {"panels": {("ResNet", "NN"): _heatmap_data("ResNet", "NN"),
                       ("ResNet+", "NN+"): _heatmap_data("ResNet+", "NN+")}}
