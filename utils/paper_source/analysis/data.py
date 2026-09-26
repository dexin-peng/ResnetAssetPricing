from __future__ import annotations

import re

import numpy as np
import pandas as pd

from utils.paper_source.data_spec import (
    PREFIX_RE as PREFIX_RE,
    FAMILY_NAMES as FAMILY_NAMES,
    MAIN_FAMILIES as MAIN_FAMILIES,
    SUPPLEMENTARY_FAMILIES as SUPPLEMENTARY_FAMILIES,
    REPORT_FAMILIES as REPORT_FAMILIES,
    PUBLIC_WIDTH_SCHEDULES as PUBLIC_WIDTH_SCHEDULES,
    PLANNED_MAX_DEPTH as PLANNED_MAX_DEPTH,
    PRIMARY_BENCHMARK_TEMPLATE as PRIMARY_BENCHMARK_TEMPLATE,
    PRIMARY_BENCHMARK_DEPTH as PRIMARY_BENCHMARK_DEPTH,
    PRIMARY_PAIRED_TEMPLATE as PRIMARY_PAIRED_TEMPLATE,
    PRIMARY_PAIRED_DEPTH as PRIMARY_PAIRED_DEPTH,
    BRIEF_PANEL as BRIEF_PANEL,
    BRIEF_ALL as BRIEF_ALL,
    DECILE_PANEL as DECILE_PANEL,
    TURNOVER_PANEL as TURNOVER_PANEL,
    COVERAGE_SUMMARY as COVERAGE_SUMMARY,
    FALLBACK_MONTH_COLUMNS as FALLBACK_MONTH_COLUMNS,
    CONSTANT_PREDICT_FALLBACK_COLUMN as CONSTANT_PREDICT_FALLBACK_COLUMN,
)





_NONFALLBACK_CELL_CACHE: dict[bool, pd.DataFrame] = {}
_NONCOLLAPSED_CELL_CACHE: dict[bool, pd.DataFrame] = {}
_DECILE_RETURNS_CACHE: dict[bool, pd.DataFrame] = {}
_TURNOVER_PANEL_CACHE: dict[bool, pd.DataFrame] = {}


def normalize_decile_label(value: object) -> object:
    if pd.isna(value):
        return value
    text = str(value).strip()
    if text.upper() == "LS":
        return "LS"
    numeric = pd.to_numeric(text, errors="coerce")
    if pd.notna(numeric) and float(numeric).is_integer():
        decile = int(numeric)
        if 1 <= decile <= 10:
            return str(decile)
    return text


def parse_prefix(prefix: str) -> dict[str, int | str] | None:
    match = PREFIX_RE.fullmatch(str(prefix))
    if not match:
        return None
    stem = match.group("stem")
    seed = int(match.group("seed"))
    if seed not in PUBLIC_WIDTH_SCHEDULES:
        return None
    return {
        "family_stem": stem,
        "family": FAMILY_NAMES[stem],
        "seed": seed,
        "depth": int(match.group("depth")),
    }


def _coerce_brief_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column in ("seed", "depth"):
        df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")
    numeric_columns = [
        "Out-of-Sample R2",
        "Value Weighted Long Short Sharpe Ratio",
        "Value Weighted LS FF5 Alpha/%",
        "Value Weighted LS FF5 t-stat",
        "Value Weighted Long Short Maximum Drawdown",
        "Value Weighted Long Short Maximum One Month Loss",
        "Value Weighted LS Turnover/%",
        "Equal Weighted LS FF5 Alpha/%",
        "Equal Weighted LS FF5 t-stat",
        "Equal Weighted Long Short Maximum Drawdown",
        "Equal Weighted Long Short Maximum One Month Loss",
        "Equal Weighted LS Turnover/%",
        "Equal Weighted Long Short Sharpe Ratio",
        "Trainable Parameters",
        *FALLBACK_MONTH_COLUMNS,
    ]
    for column in numeric_columns:
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _fallback_mask_from_columns(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    for column in FALLBACK_MONTH_COLUMNS:
        if column in df.columns:
            mask |= pd.to_numeric(df[column], errors="coerce").fillna(0.0) > 0.0
    return mask


def _constant_fallback_mask_from_columns(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    if CONSTANT_PREDICT_FALLBACK_COLUMN in df.columns:
        mask |= pd.to_numeric(df[CONSTANT_PREDICT_FALLBACK_COLUMN], errors="coerce").fillna(0.0) > 0.0
    return mask


def _filter_public_width_schedules(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "seed" not in df.columns:
        return df.copy()
    seeds = pd.to_numeric(df["seed"], errors="coerce")
    return df.loc[seeds.isin(PUBLIC_WIDTH_SCHEDULES)].copy()


def _read_brief_panel() -> pd.DataFrame:
    if not BRIEF_PANEL.exists():
        raise FileNotFoundError(f"Missing {BRIEF_PANEL}. Run python -m utils.paper_source.collect first.")
    return _filter_public_width_schedules(_coerce_brief_columns(pd.read_csv(BRIEF_PANEL)))


def _read_brief_results() -> pd.DataFrame:
    if not BRIEF_ALL.exists():
        raise FileNotFoundError(f"Missing {BRIEF_ALL}. Run python -m utils.paper_source.collect first.")
    return _filter_public_width_schedules(_coerce_brief_columns(pd.read_csv(BRIEF_ALL)))


def _nonfallback_cell_frame(*, include_group: bool) -> pd.DataFrame:
    if include_group in _NONFALLBACK_CELL_CACHE:
        return _NONFALLBACK_CELL_CACHE[include_group].copy()
    ref = _read_brief_panel()
    key_cols = ["family", "seed", "depth"]
    if include_group and "group" in ref.columns:
        key_cols = ["group"] + key_cols
    ref = ref.loc[~_fallback_mask_from_columns(ref), key_cols].copy()
    ref = ref.dropna(subset=["family", "seed", "depth"])
    ref["seed"] = pd.to_numeric(ref["seed"], errors="coerce").astype(int)
    ref["depth"] = pd.to_numeric(ref["depth"], errors="coerce").astype(int)
    ref = ref.drop_duplicates()
    _NONFALLBACK_CELL_CACHE[include_group] = ref
    return ref.copy()


def _noncollapsed_cell_frame(*, include_group: bool) -> pd.DataFrame:
    if include_group in _NONCOLLAPSED_CELL_CACHE:
        return _NONCOLLAPSED_CELL_CACHE[include_group].copy()
    ref = _read_brief_panel()
    key_cols = ["family", "seed", "depth"]
    if include_group and "group" in ref.columns:
        key_cols = ["group"] + key_cols
    ref = ref.loc[~_constant_fallback_mask_from_columns(ref), key_cols].copy()
    ref = ref.dropna(subset=["family", "seed", "depth"])
    ref["seed"] = pd.to_numeric(ref["seed"], errors="coerce").astype(int)
    ref["depth"] = pd.to_numeric(ref["depth"], errors="coerce").astype(int)
    ref = ref.drop_duplicates()
    _NONCOLLAPSED_CELL_CACHE[include_group] = ref
    return ref.copy()


def _filter_nonfallback_rows(df: pd.DataFrame) -> pd.DataFrame:
    if any(column in df.columns for column in FALLBACK_MONTH_COLUMNS):
        return df.loc[~_fallback_mask_from_columns(df)].copy()

    if not {"family", "seed", "depth"}.issubset(df.columns):
        return df.copy()
    include_group = "group" in df.columns
    key_cols = ["family", "seed", "depth"]
    if include_group:
        key_cols = ["group"] + key_cols

    valid_cells = _nonfallback_cell_frame(include_group=include_group)
    if valid_cells.empty:
        return df.iloc[0:0].copy()

    out = df.copy()
    out["_rowid_filter"] = np.arange(len(out))
    out["_seed_filter"] = pd.to_numeric(out["seed"], errors="coerce")
    out["_depth_filter"] = pd.to_numeric(out["depth"], errors="coerce")
    valid = out[["family", "_seed_filter", "_depth_filter"]].notna().all(axis=1)
    if not valid.any():
        return out.drop(columns=["_rowid_filter", "_seed_filter", "_depth_filter"]).iloc[0:0].copy()

    probe_cols = ["_rowid_filter"] + key_cols
    probe = out.loc[valid, probe_cols].copy()
    probe["seed"] = pd.to_numeric(probe["seed"], errors="coerce").astype(int)
    probe["depth"] = pd.to_numeric(probe["depth"], errors="coerce").astype(int)
    paired_ids = probe.merge(valid_cells, on=key_cols, how="inner")["_rowid_filter"]
    keep = out["_rowid_filter"].isin(paired_ids)
    return out.loc[keep].drop(columns=["_rowid_filter", "_seed_filter", "_depth_filter"]).copy()


def filter_noncollapsed_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep cells that do not collapse to constant forecasts.

    Seeded tie-split fallback months are retained because forecasts still have
    cross-sectional information; constant-predict fallback cells are excluded.
    """
    if CONSTANT_PREDICT_FALLBACK_COLUMN in df.columns:
        return df.loc[~_constant_fallback_mask_from_columns(df)].copy()

    if not {"family", "seed", "depth"}.issubset(df.columns):
        return df.copy()
    include_group = "group" in df.columns
    key_cols = ["family", "seed", "depth"]
    if include_group:
        key_cols = ["group"] + key_cols

    valid_cells = _noncollapsed_cell_frame(include_group=include_group)
    if valid_cells.empty:
        return df.iloc[0:0].copy()

    out = df.copy()
    out["_rowid_filter"] = np.arange(len(out))
    out["_seed_filter"] = pd.to_numeric(out["seed"], errors="coerce")
    out["_depth_filter"] = pd.to_numeric(out["depth"], errors="coerce")
    valid = out[["family", "_seed_filter", "_depth_filter"]].notna().all(axis=1)
    if not valid.any():
        return out.drop(columns=["_rowid_filter", "_seed_filter", "_depth_filter"]).iloc[0:0].copy()

    probe_cols = ["_rowid_filter"] + key_cols
    probe = out.loc[valid, probe_cols].copy()
    probe["seed"] = pd.to_numeric(probe["seed"], errors="coerce").astype(int)
    probe["depth"] = pd.to_numeric(probe["depth"], errors="coerce").astype(int)
    paired_ids = probe.merge(valid_cells, on=key_cols, how="inner")["_rowid_filter"]
    keep = out["_rowid_filter"].isin(paired_ids)
    return out.loc[keep].drop(columns=["_rowid_filter", "_seed_filter", "_depth_filter"]).copy()


def load_brief_panel(*, include_fallback: bool = False) -> pd.DataFrame:
    df = _read_brief_panel()
    if include_fallback:
        return df
    return _filter_nonfallback_rows(df)


def load_brief_results(*, include_fallback: bool = True) -> pd.DataFrame:
    df = _read_brief_results()
    if include_fallback:
        return df
    return _filter_nonfallback_rows(df)


def load_decile_returns(*, include_fallback: bool = False) -> pd.DataFrame:
    if include_fallback in _DECILE_RETURNS_CACHE:
        return _DECILE_RETURNS_CACHE[include_fallback].copy(deep=False)
    if not DECILE_PANEL.exists():
        raise FileNotFoundError(f"Missing {DECILE_PANEL}. Run python -m utils.paper_source.collect first.")
    paths = sorted(DECILE_PANEL.glob("part-*.csv.gz")) if DECILE_PANEL.is_dir() else [DECILE_PANEL]
    if not paths:
        raise FileNotFoundError(f"No monthly portfolio inputs in {DECILE_PANEL}")
    df = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    df["decile"] = df["decile"].map(normalize_decile_label)
    df["DATE"] = pd.to_datetime(df["DATE"])
    df["return"] = pd.to_numeric(df["return"], errors="coerce")
    for column in ("seed", "depth"):
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df = _filter_public_width_schedules(df)
    if not include_fallback:
        df = _filter_nonfallback_rows(df)
    _DECILE_RETURNS_CACHE[include_fallback] = df
    return df.copy(deep=False)


def load_turnover_panel(*, include_fallback: bool = False) -> pd.DataFrame:
    if include_fallback in _TURNOVER_PANEL_CACHE:
        return _TURNOVER_PANEL_CACHE[include_fallback].copy(deep=False)
    if not TURNOVER_PANEL.exists():
        raise FileNotFoundError(f"Missing {TURNOVER_PANEL}. Rerun econ and python -m utils.paper_source.collect to collect monthly turnover.")
    df = pd.read_csv(TURNOVER_PANEL)
    if df.empty:
        raise ValueError(f"{TURNOVER_PANEL} is empty. Rerun econ with monthly turnover export enabled.")
    df["DATE"] = pd.to_datetime(df["DATE"])
    df["turnover"] = pd.to_numeric(df["turnover"], errors="coerce")
    for column in ("seed", "depth"):
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df = _filter_public_width_schedules(df)
    if not include_fallback:
        df = _filter_nonfallback_rows(df)
    _TURNOVER_PANEL_CACHE[include_fallback] = df
    return df.copy(deep=False)


def completed_results(
    families: tuple[str, ...] = REPORT_FAMILIES,
    *,
    include_fallback: bool = False,
) -> pd.DataFrame:
    df = load_brief_results(include_fallback=include_fallback)
    return df[df["family"].isin(families)].copy()
