"""Table data analysis. No asset rendering or generator invocation."""
from __future__ import annotations
import numpy as np
import pandas as pd
from utils.paper_source.artifact_cache import Estimate
from .data import (
    REPORT_FAMILIES,
    completed_results,
    filter_noncollapsed_rows,
    load_brief_panel,
    load_decile_returns,
    load_turnover_panel,
    normalize_decile_label,
)
from utils.paper_source.paths import SOURCE_DIR
from utils.paper_source.table_spec import (
    BASELINE_DEPTH_BANDS,
    DEPTH_CONTRAST_BOOTSTRAP_SEEDS,
    DECILE_PORTFOLIO_STATISTICS_SOURCE,
    DEPTH_BANDS,
    EW_SR,
    EW_TURNOVER,
    FF5_FACTOR_PATHS,
    FORECAST_LEVEL_R2_COLUMN,
    FORECAST_LEVEL_R2_SOURCE,
    FORECAST_LEVEL_SCALE_COLUMN,
    INITIAL_LONG_SHORT_GROSS_TURNOVER,
    LW_T_COL,
    MARKET_STATE_INFERENCE_MIN_MONTHS,
    NBER_BUSINESS_CYCLE_PEAK_TROUGH,
    OOS_R2,
    PAIR_METRIC_COLUMNS,
    PREDICTOR_LIST_SOURCE,
    RAW_RESULT_COLUMNS,
    REPORT_MODEL_ORDER,
    SR_BOOTSTRAP_SEEDS,
    TABLE_R2_OOS,
    TABLE_R2_SCALE,
    VW_ALPHA,
    VW_ALPHA_T,
    VW_SR,
    VW_TURNOVER,
)
from utils.paper_source.table_labels import (
    _width_schedule_label,
    _sr_stars,
    _tstat_stars,
)


def _paired_family_pair(left_family: str, right_family: str) -> pd.DataFrame:
    df = filter_noncollapsed_rows(completed_results((left_family, right_family), include_fallback=True))
    keep = ["family", "seed", "depth", "Asset Prefix"] + [c for c in PAIR_METRIC_COLUMNS if c in df.columns]
    wide = df[keep].pivot(index=["seed", "depth"], columns="family")
    if left_family not in wide.columns.get_level_values(1) or right_family not in wide.columns.get_level_values(1):
        return pd.DataFrame()
    rows = []
    for seed, depth in wide.index:
        row = {"seed": int(seed), "depth": int(depth)}
        missing = False
        for metric in PAIR_METRIC_COLUMNS:
            if (metric, left_family) not in wide or (metric, right_family) not in wide:
                continue
            row[f"{left_family} {metric}"] = wide.loc[(seed, depth), (metric, left_family)]
            row[f"{right_family} {metric}"] = wide.loc[(seed, depth), (metric, right_family)]
        for family in (left_family, right_family):
            if ("Asset Prefix", family) not in wide or pd.isna(wide.loc[(seed, depth), ("Asset Prefix", family)]):
                missing = True
        if not missing:
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["seed", "depth"]).reset_index(drop=True)


def _load_ff5_factors() -> pd.DataFrame:
    for path in FF5_FACTOR_PATHS:
        if path.exists():
            factors = pd.read_csv(path)
            break
    else:
        raise FileNotFoundError("Missing FF5 factors for Table 4 alpha estimation.")
    date = factors["DATE"].astype(str)
    factors = factors.copy()
    factors.index = pd.to_datetime(date, format="%Y%m") + pd.offsets.MonthEnd(0)
    factors = factors[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]].apply(pd.to_numeric, errors="coerce")
    if factors.abs().to_numpy(dtype=float).max() > 1.0:
        factors = factors / 100.0
    factors.columns = ["Mkt_RF", "SMB", "HML", "RMW", "CMA"]
    return factors.dropna()


def _hac_alpha(y: np.ndarray, x: np.ndarray, lags: int = 12) -> tuple[float, float]:
    if y.shape[0] <= x.shape[1] + 1:
        return np.nan, np.nan
    xtx_inv = np.linalg.pinv(x.T @ x)
    beta = xtx_inv @ (x.T @ y)
    resid = y - x @ beta
    xu = x * resid[:, None]
    s = xu.T @ xu
    max_lag = min(int(lags), y.shape[0] - 1)
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (lags + 1.0)
        s += weight * (xu[lag:].T @ xu[:-lag] + xu[:-lag].T @ xu[lag:])
    var_beta = xtx_inv @ s @ xtx_inv
    se = np.sqrt(np.maximum(np.diag(var_beta), 0.0))
    alpha = float(beta[0])
    t_alpha = float(alpha / se[0]) if se[0] else np.nan
    return alpha, t_alpha


def _ls_return_lookup(
    decile: pd.DataFrame,
    *,
    families: tuple[str, ...],
    group: str = "ALL",
    weighting: str = "VW",
) -> dict[tuple[str, int, int], pd.Series]:
    part = decile[
        (decile["family"].isin(families))
        & (decile["group"] == group)
        & (decile["weighting"] == weighting)
        & (decile["decile"] == "LS")
    ].copy()
    if part.empty:
        return {}
    part["seed"] = pd.to_numeric(part["seed"], errors="coerce")
    part["depth"] = pd.to_numeric(part["depth"], errors="coerce")
    part["return"] = pd.to_numeric(part["return"], errors="coerce")
    part = part.dropna(subset=["seed", "depth", "return"])
    lookup: dict[tuple[str, int, int], pd.Series] = {}
    for (family, seed, depth), block in part.groupby(["family", "seed", "depth"], sort=False):
        lookup[(str(family), int(seed), int(depth))] = block.sort_values("DATE").set_index("DATE")["return"]
    return lookup


def _cell_averaged_delta_ls_return(
    ls_returns: dict[tuple[str, int, int], pd.Series],
    cells: pd.DataFrame,
    *,
    left_family: str,
    right_family: str,
) -> pd.Series:
    diff_series = []
    for _, row in cells.iterrows():
        seed = int(row["seed"])
        depth = int(row["depth"])
        left = ls_returns.get((left_family, seed, depth), pd.Series(dtype=float))
        right = ls_returns.get((right_family, seed, depth), pd.Series(dtype=float))
        joined = pd.concat([left.rename("left"), right.rename("right")], axis=1, join="inner").dropna()
        if joined.empty:
            continue
        diff_series.append((joined["left"] - joined["right"]).rename(f"s{seed}d{depth}"))
    if not diff_series:
        return pd.Series(dtype=float)
    return pd.concat(diff_series, axis=1).mean(axis=1).dropna().sort_index()


def _cell_averaged_return_from_lookup(
    returns: dict[tuple[str, int, int], pd.Series],
    *,
    family: str,
    cells: list[tuple[int, int]],
) -> pd.Series:
    cell_series = []
    for seed, depth in cells:
        ret = returns.get((family, int(seed), int(depth)), pd.Series(dtype=float))
        if ret.empty:
            continue
        cell_series.append(ret.rename(f"s{int(seed)}d{int(depth)}"))
    if not cell_series:
        return pd.Series(dtype=float)
    return pd.concat(cell_series, axis=1).mean(axis=1).dropna().sort_index()


def _ff5_alpha_for_return(return_series: pd.Series, factors: pd.DataFrame) -> tuple[float, float]:
    panel = pd.concat([return_series.rename("return"), factors], axis=1, join="inner").dropna()
    if panel.empty:
        return np.nan, np.nan
    y = panel["return"].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(panel)), panel[["Mkt_RF", "SMB", "HML", "RMW", "CMA"]].to_numpy(dtype=float)])
    alpha, t_alpha = _hac_alpha(y, x)
    return alpha * 100.0, t_alpha


def _ff5_alpha_for_delta_return(delta_return: pd.Series, factors: pd.DataFrame) -> tuple[float, float]:
    return _ff5_alpha_for_return(delta_return, factors)


def _circular_block_indices(rng: np.random.Generator, n: int, block_length: int) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=int)
    block_length = max(1, min(int(block_length), n))
    starts = rng.integers(0, n, size=int(np.ceil(n / block_length)))
    blocks = [(start + np.arange(block_length)) % n for start in starts]
    return np.concatenate(blocks)[:n].astype(int)


def _lw_sharpe_delta_components(left: np.ndarray, right: np.ndarray) -> dict[str, object]:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    valid = np.isfinite(left) & np.isfinite(right)
    left = left[valid]
    right = right[valid]
    n = int(len(left))
    if n <= 2:
        return {}
    mu_left = float(left.mean())
    mu_right = float(right.mean())
    sigma_left = float(left.std(ddof=1))
    sigma_right = float(right.std(ddof=1))
    if sigma_left <= 0.0 or sigma_right <= 0.0:
        return {}
    delta = np.sqrt(12.0) * (mu_left / sigma_left - mu_right / sigma_right)
    centered_left = left - mu_left
    centered_right = right - mu_right
    grad = np.sqrt(12.0) * np.array(
        [
            1.0 / sigma_left,
            -1.0 / sigma_right,
            -0.5 * mu_left / (sigma_left**3),
            0.5 * mu_right / (sigma_right**3),
        ],
        dtype=float,
    )
    moments = np.column_stack(
        [
            centered_left,
            centered_right,
            centered_left**2 - sigma_left**2,
            centered_right**2 - sigma_right**2,
        ]
    )
    influence = moments @ grad
    return {"delta": float(delta), "influence": influence, "months": n}


def _hac_se_from_influence(influence: np.ndarray, *, lags: int = 12) -> float:
    values = np.asarray(influence, dtype=float)
    values = values[np.isfinite(values)]
    n = int(len(values))
    if n <= 2:
        return np.nan
    centered = values - values.mean()
    max_lag = min(int(lags), n - 1)
    long_run_var = float(np.dot(centered, centered) / n)
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        gamma = float(np.dot(centered[lag:], centered[:-lag]) / n)
        long_run_var += 2.0 * weight * gamma
    if not np.isfinite(long_run_var) or long_run_var <= 0.0:
        return np.nan
    return float(np.sqrt(long_run_var / n))


def _block_se_from_influence(influence: np.ndarray, *, block_length: int = 12) -> float:
    values = np.asarray(influence, dtype=float)
    values = values[np.isfinite(values)]
    n = int(len(values))
    if n <= 2:
        return np.nan
    block_length = max(1, min(int(block_length), n))
    n_blocks = n // block_length
    if n_blocks <= 1:
        return _hac_se_from_influence(values, lags=block_length)
    trimmed = values[: n_blocks * block_length].reshape(n_blocks, block_length)
    block_scores = trimmed.sum(axis=1) / np.sqrt(block_length)
    long_run_var = float(np.dot(block_scores, block_scores) / n_blocks)
    if not np.isfinite(long_run_var) or long_run_var <= 0.0:
        return np.nan
    return float(np.sqrt(long_run_var / n))


def _average_influence_by_date(components: list[dict[str, object]]) -> np.ndarray:
    series = []
    for component in components:
        dates = component.get("dates")
        influence = np.asarray(component["influence"], dtype=float)
        if dates is None or len(dates) != len(influence):
            series.append(pd.Series(influence))
        else:
            series.append(pd.Series(influence, index=pd.Index(dates)))
    if not series:
        return np.asarray([], dtype=float)
    return pd.concat(series, axis=1).mean(axis=1).dropna().to_numpy(dtype=float)


def _average_influence_by_position(influences: list[np.ndarray]) -> np.ndarray:
    if not influences:
        return np.asarray([], dtype=float)
    max_len = max(len(influence) for influence in influences)
    if max_len == 0:
        return np.asarray([], dtype=float)
    matrix = np.full((max_len, len(influences)), np.nan, dtype=float)
    for col, influence in enumerate(influences):
        values = np.asarray(influence, dtype=float)
        matrix[: len(values), col] = values
    with np.errstate(invalid="ignore"):
        return np.nanmean(matrix, axis=1)


def _lw_components_for_specs(specs: list[dict[str, object]]) -> list[dict[str, object]]:
    components: list[dict[str, object]] = []
    for spec in specs:
        component = _lw_sharpe_delta_components(spec["left"], spec["right"])
        if not component:
            continue
        component["dates"] = spec.get("dates")
        components.append(component)
    return components


def _ledoit_wolf_sr_difference_test(
    specs: list[dict[str, object]],
    *,
    seed: int,
    n_boot: int = 1000,
    block_length: int = 12,
    hac_lags: int = 12,
) -> dict[str, object]:
    components = _lw_components_for_specs(specs)
    deltas = [float(component["delta"]) for component in components if np.isfinite(float(component["delta"]))]
    if not deltas:
        return {}
    delta_hat = float(np.nanmean(deltas))
    influence = _average_influence_by_date(components)
    se_hat = _hac_se_from_influence(influence, lags=hac_lags)
    if not np.isfinite(se_hat) or se_hat <= 0.0:
        return {}

    rng = np.random.default_rng(seed)
    index_cache: dict[int, np.ndarray] = {
        int(n): np.vstack([_circular_block_indices(rng, int(n), block_length) for _ in range(n_boot)])
        for n in sorted({int(len(spec["left"])) for spec in specs})
    }
    boot_t: list[float] = []
    for boot_idx in range(n_boot):
        boot_deltas: list[float] = []
        boot_influences: list[np.ndarray] = []
        for spec in specs:
            idx = index_cache[int(len(spec["left"]))][boot_idx]
            component = _lw_sharpe_delta_components(spec["left"][idx], spec["right"][idx])
            if not component:
                continue
            boot_deltas.append(float(component["delta"]))
            boot_influences.append(np.asarray(component["influence"], dtype=float))
        if not boot_deltas:
            continue
        boot_delta = float(np.nanmean(boot_deltas))
        boot_influence = _average_influence_by_position(boot_influences)
        boot_se = _block_se_from_influence(boot_influence, block_length=block_length)
        if np.isfinite(boot_se) and boot_se > 0.0:
            boot_t.append((boot_delta - delta_hat) / boot_se)
    boot_t_array = np.asarray(boot_t, dtype=float)
    boot_t_array = boot_t_array[np.isfinite(boot_t_array)]
    if boot_t_array.size == 0:
        return {}
    observed_t = float(delta_hat / se_hat)
    critical = float(np.nanquantile(np.abs(boot_t_array), 0.95))
    p_value = float((np.sum(np.abs(boot_t_array) >= abs(observed_t)) + 1.0) / (len(boot_t_array) + 1.0))
    return {
        "delta": delta_hat,
        "se": se_hat,
        "ci_low": delta_hat - critical * se_hat,
        "ci_high": delta_hat + critical * se_hat,
        "p_value": p_value,
        "specs": int(len(deltas)),
        "months": int(min(component["months"] for component in components)),
    }


def _safe_tstat_from_se(estimate: object, se: object) -> float:
    if estimate is None or se is None or pd.isna(estimate) or pd.isna(se):
        return np.nan
    estimate_value = float(estimate)
    se_value = float(se)
    if not np.isfinite(estimate_value) or not np.isfinite(se_value) or se_value <= 0.0:
        return np.nan
    return estimate_value / se_value


def _paired_ls_specs(
    decile: pd.DataFrame,
    *,
    left_family: str,
    right_family: str,
    pairs: pd.DataFrame,
    group: str = "ALL",
    weighting: str = "VW",
) -> list[dict[str, object]]:
    series_by_key = _ls_return_lookup(
        decile,
        families=(left_family, right_family),
        group=group,
        weighting=weighting,
    )
    return _paired_return_specs_from_lookup(
        series_by_key,
        left_family=left_family,
        right_family=right_family,
        pairs=pairs,
    )


def _paired_return_specs_from_lookup(
    series_by_key: dict[tuple[str, int, int], pd.Series],
    *,
    left_family: str,
    right_family: str,
    pairs: pd.DataFrame,
    min_months: int = 24,
) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    for _, row in pairs.sort_values(["seed", "depth"]).iterrows():
        width = int(row["seed"])
        depth = int(row["depth"])
        left = series_by_key.get((left_family, width, depth), pd.Series(dtype=float))
        right = series_by_key.get((right_family, width, depth), pd.Series(dtype=float))
        joined = pd.concat([left.rename("left"), right.rename("right")], axis=1, join="inner").dropna()
        if len(joined) <= int(min_months):
            continue
        specs.append(
            {
                "width": width,
                "depth": depth,
                "left": joined["left"].to_numpy(dtype=float),
                "right": joined["right"].to_numpy(dtype=float),
                "dates": joined.index.to_numpy(),
                "months": int(len(joined)),
            }
        )
    return specs


def _vw_ls_cell_stats(decile: pd.DataFrame) -> pd.DataFrame:
    part = decile[
        (decile["group"] == "ALL")
        & (decile["weighting"] == "VW")
        & (decile["decile"].astype(str) == "LS")
    ].copy()
    if part.empty:
        return pd.DataFrame(columns=["family", "seed", "depth", "VW mean", "VW vol", "VW SR from returns", "Months"])
    part["seed"] = pd.to_numeric(part["seed"], errors="coerce")
    part["depth"] = pd.to_numeric(part["depth"], errors="coerce")
    part["return"] = pd.to_numeric(part["return"], errors="coerce")
    part = part.dropna(subset=["seed", "depth", "return"])
    stats = (
        part.groupby(["family", "seed", "depth"])["return"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    stats["seed"] = stats["seed"].astype(int)
    stats["depth"] = stats["depth"].astype(int)
    stats["VW mean"] = stats["mean"] * 100.0
    stats["VW vol"] = stats["std"] * 100.0
    stats["VW SR from returns"] = np.where(
        stats["std"] > 0.0,
        stats["mean"] / stats["std"] * np.sqrt(12.0),
        np.nan,
    )
    stats["Months"] = stats["count"].astype(int)
    return stats[["family", "seed", "depth", "VW mean", "VW vol", "VW SR from returns", "Months"]]


def _baseline_mean_row(part: pd.DataFrame, *, depth_group: str) -> dict[str, object]:
    def mean_of(column: str) -> float:
        return _cell_weighted_mean(part, column)

    return {
        "Model": part.iloc[0]["family"],
        "Depth Group": depth_group,
        TABLE_R2_OOS: mean_of(TABLE_R2_OOS),
        "VW Mean": mean_of("VW mean"),
        "VW Vol.": mean_of("VW vol"),
        "VW SR": mean_of("VW SR table"),
        "EW SR": mean_of(EW_SR),
        "FF5 $\\alpha$": mean_of(VW_ALPHA),
        "$t$(FF5)": mean_of(VW_ALPHA_T),
        "Turnover": mean_of(VW_TURNOVER),
    }


def _load_forecast_level_r2() -> pd.DataFrame:
    if not FORECAST_LEVEL_R2_SOURCE.exists():
        return pd.DataFrame(columns=["family", "seed", "depth", TABLE_R2_OOS, TABLE_R2_SCALE])
    source = pd.read_csv(FORECAST_LEVEL_R2_SOURCE)
    required = {"family", "seed", "depth", FORECAST_LEVEL_R2_COLUMN}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(
            f"{FORECAST_LEVEL_R2_SOURCE} is missing required columns: {sorted(missing)}"
        )
    columns = ["family", "seed", "depth", FORECAST_LEVEL_R2_COLUMN]
    if FORECAST_LEVEL_SCALE_COLUMN in source.columns:
        columns.append(FORECAST_LEVEL_SCALE_COLUMN)
    out = source[columns].copy()
    out["seed"] = pd.to_numeric(out["seed"], errors="coerce")
    out["depth"] = pd.to_numeric(out["depth"], errors="coerce")
    out[TABLE_R2_OOS] = pd.to_numeric(out[FORECAST_LEVEL_R2_COLUMN], errors="coerce")
    out[TABLE_R2_SCALE] = (
        pd.to_numeric(out[FORECAST_LEVEL_SCALE_COLUMN], errors="coerce")
        if FORECAST_LEVEL_SCALE_COLUMN in out.columns
        else np.nan
    )
    out = out.dropna(subset=["family", "seed", "depth"])
    out["seed"] = out["seed"].astype(int)
    out["depth"] = out["depth"].astype(int)
    return out[["family", "seed", "depth", TABLE_R2_OOS, TABLE_R2_SCALE]]


def _attach_forecast_level_r2(
    pairs: pd.DataFrame,
    *,
    left_family: str,
    right_family: str,
    source: pd.DataFrame,
) -> pd.DataFrame:
    out = pairs.copy()
    for family in (left_family, right_family):
        lookup: dict[tuple[int, int], float] = {}
        part = source[source["family"] == family]
        for _, row in part.iterrows():
            if pd.isna(row.get("seed")) or pd.isna(row.get("depth")):
                continue
            lookup[(int(row["seed"]), int(row["depth"]))] = row.get(TABLE_R2_OOS, np.nan)
        out[f"{family} {TABLE_R2_OOS}"] = [
            lookup.get((int(row["seed"]), int(row["depth"])), np.nan)
            for _, row in out.iterrows()
        ]
    return out


def _sr_lw_seed(left_family: str, right_family: str, depth_label: str, *, offset: int = 0) -> int:
    seed = SR_BOOTSTRAP_SEEDS.get((left_family, right_family), {}).get(depth_label)
    if seed is None:
        family_code = sum(ord(char) for char in f"{left_family}:{right_family}:{depth_label}")
        seed = 20260000 + family_code
    return int(seed) + int(offset)


def _sr_ledoit_wolf_test(
    specs: list[dict[str, object]],
    *,
    seed: int,
    n_boot: int = 1000,
    block_length: int = 12,
) -> dict[str, object]:
    test = _ledoit_wolf_sr_difference_test(
        specs,
        seed=seed,
        n_boot=n_boot,
        block_length=block_length,
        hac_lags=block_length,
    )
    return test if test else {}


def _weighted_group_sr_contrast_test(
    return_panel: pd.DataFrame,
    *,
    shallow_columns: list[tuple[int, int]],
    deep_columns: list[tuple[int, int]],
    seed: int,
    n_boot: int = 1000,
    block_length: int = 12,
) -> dict[str, object]:
    columns = shallow_columns + deep_columns
    panel = return_panel.reindex(columns=pd.MultiIndex.from_tuples(columns)).dropna()
    if len(panel) <= 24 or not shallow_columns or not deep_columns:
        return {}
    values = panel.to_numpy(dtype=float)
    weights = np.concatenate(
        [
            np.full(len(shallow_columns), -1.0 / len(shallow_columns)),
            np.full(len(deep_columns), 1.0 / len(deep_columns)),
        ]
    )

    def components(sample: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        means = sample.mean(axis=0)
        sigmas = sample.std(axis=0, ddof=1)
        if np.any(~np.isfinite(sigmas)) or np.any(sigmas <= 0.0):
            return np.nan, np.asarray([], dtype=float), np.asarray([], dtype=float)
        sharpes = np.sqrt(12.0) * means / sigmas
        centered = sample - means
        influences = np.sqrt(12.0) * (
            centered / sigmas
            - means * (centered**2 - sigmas**2) / (2.0 * sigmas**3)
        )
        return float(sharpes @ weights), influences @ weights, sharpes

    delta_hat, influence_hat, sharpes_hat = components(values)
    se_hat = _hac_se_from_influence(influence_hat, lags=block_length)
    if not np.isfinite(delta_hat) or not np.isfinite(se_hat) or se_hat <= 0.0:
        return {}

    rng = np.random.default_rng(seed)
    indices = np.vstack(
        [_circular_block_indices(rng, len(panel), block_length) for _ in range(n_boot)]
    )
    boot_t: list[float] = []
    for draw_indices in indices:
        boot_delta, boot_influence, _boot_sharpes = components(values[draw_indices])
        boot_se = _block_se_from_influence(boot_influence, block_length=block_length)
        if np.isfinite(boot_delta) and np.isfinite(boot_se) and boot_se > 0.0:
            boot_t.append((boot_delta - delta_hat) / boot_se)
    boot_t_array = np.asarray(boot_t, dtype=float)
    boot_t_array = boot_t_array[np.isfinite(boot_t_array)]
    if boot_t_array.size == 0:
        return {}
    observed_t = float(delta_hat / se_hat)
    critical = float(np.nanquantile(np.abs(boot_t_array), 0.95))
    p_value = float(
        (np.sum(np.abs(boot_t_array) >= abs(observed_t)) + 1.0)
        / (len(boot_t_array) + 1.0)
    )
    shallow_sr = float(np.mean(sharpes_hat[: len(shallow_columns)]))
    deep_sr = float(np.mean(sharpes_hat[len(shallow_columns) :]))
    return {
        "delta": delta_hat,
        "se": se_hat,
        "t": observed_t,
        "p_value": p_value,
        "ci_low": delta_hat - critical * se_hat,
        "ci_high": delta_hat + critical * se_hat,
        "shallow_sr": shallow_sr,
        "deep_sr": deep_sr,
        "shallow_cells": int(len(shallow_columns)),
        "deep_cells": int(len(deep_columns)),
        "months": int(len(panel)),
        "bootstrap_draws": int(len(boot_t_array)),
    }


def _cell_weighted_mean(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return np.nan
    return float(values.mean())


def _cell_weighted_mean_from_values(values: list[float]) -> float:
    if not values:
        return np.nan
    values_array = np.asarray(values, dtype=float)
    values_array = values_array[np.isfinite(values_array)]
    if values_array.size == 0:
        return np.nan
    return float(np.nanmean(values_array))


def _raw_decile_summary(decile: pd.DataFrame, families: tuple[str, ...]) -> pd.DataFrame:
    labels = tuple(str(i) for i in range(1, 11))
    decile = decile.copy()
    decile["decile"] = decile["decile"].map(normalize_decile_label)
    part = decile[
        (decile["family"].isin(families))
        & (decile["group"] == "ALL")
        & (decile["weighting"] == "VW")
        & (decile["decile"].astype(str).isin(labels))
    ].copy()
    if part.empty:
        return pd.DataFrame(columns=["family", "seed", "depth", "Months", *[f"D{i}" for i in range(1, 11)]])
    part["seed"] = pd.to_numeric(part["seed"], errors="coerce")
    part["depth"] = pd.to_numeric(part["depth"], errors="coerce")
    part["decile_num"] = pd.to_numeric(part["decile"], errors="coerce")
    part["return"] = pd.to_numeric(part["return"], errors="coerce")
    part = part.dropna(subset=["seed", "depth", "decile_num"])
    part["seed"] = part["seed"].astype(int)
    part["depth"] = part["depth"].astype(int)
    part["decile_num"] = part["decile_num"].astype(int)
    index_cols = ["family", "seed", "depth"]
    means = part.groupby(index_cols + ["decile_num"])["return"].mean().unstack("decile_num") * 100.0
    means = means.rename(columns={i: f"D{i}" for i in range(1, 11)})
    months = part.groupby(index_cols)["DATE"].nunique().rename("Months")
    out = means.join(months, how="outer").reset_index()
    for decile_id in range(1, 11):
        col = f"D{decile_id}"
        if col not in out:
            out[col] = np.nan
    return out


def _raw_result_table(families: tuple[str, ...]) -> pd.DataFrame:
    brief = filter_noncollapsed_rows(completed_results(families, include_fallback=True))
    if brief.empty:
        return pd.DataFrame(columns=RAW_RESULT_COLUMNS)
    if "group" in brief.columns:
        brief = brief[brief["group"] == "ALL"].copy()
    brief["seed"] = pd.to_numeric(brief["seed"], errors="coerce")
    brief["depth"] = pd.to_numeric(brief["depth"], errors="coerce")
    brief = brief.dropna(subset=["seed", "depth"])
    brief["seed"] = brief["seed"].astype(int)
    brief["depth"] = brief["depth"].astype(int)
    forecast_r2 = _load_forecast_level_r2()
    brief = brief.merge(forecast_r2, on=["family", "seed", "depth"], how="left")
    decile_summary = _raw_decile_summary(filter_noncollapsed_rows(load_decile_returns(include_fallback=True)), families)
    merged = brief.merge(decile_summary, on=["family", "seed", "depth"], how="left")
    family_rank = {family: idx for idx, family in enumerate(families)}
    merged["_family_rank"] = merged["family"].map(family_rank)
    merged = merged.sort_values(["seed", "depth", "_family_rank"], kind="mergesort")
    rows: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        item: dict[str, object] = {
            "Width": _width_schedule_label(row["seed"]),
            "Depth": int(row["depth"]),
            "Model": row["family"],
            "Spec": row.get("Asset Prefix", ""),
            "Months": int(row["Months"]) if pd.notna(row.get("Months", np.nan)) else np.nan,
            "$R^2_{OOS}$": row.get(TABLE_R2_OOS, row.get(OOS_R2, np.nan)),
            "Scale": row.get(TABLE_R2_SCALE, np.nan),
            "VW SR": row.get(VW_SR, np.nan),
            "FF5 $\\alpha$": row.get(VW_ALPHA, np.nan),
            "$t(\\alpha)$": row.get(VW_ALPHA_T, np.nan),
            "MDD": abs(row.get("Value Weighted Long Short Maximum Drawdown", np.nan))
            if pd.notna(row.get("Value Weighted Long Short Maximum Drawdown", np.nan))
            else np.nan,
            "Max loss": row.get("Value Weighted Long Short Maximum One Month Loss", np.nan),
            "Turnover": row.get(VW_TURNOVER, np.nan),
        }
        for decile_id in range(1, 11):
            item[f"D{decile_id}"] = row.get(f"D{decile_id}", np.nan)
        item["H-L"] = (
            item["D10"] - item["D1"]
            if pd.notna(item.get("D10", np.nan)) and pd.notna(item.get("D1", np.nan))
            else np.nan
        )
        rows.append(item)
    return pd.DataFrame(rows, columns=RAW_RESULT_COLUMNS)


def _completed_decile_cells(
    decile: pd.DataFrame,
    *,
    left_family: str,
    right_family: str | None = None,
    deciles: tuple[str, ...] = tuple(str(i) for i in range(1, 11)),
) -> list[tuple[int, int]]:
    labels = tuple(str(label) for label in deciles)
    part = decile[
        (decile["family"].isin([left_family, right_family]))
        & (decile["group"] == "ALL")
        & (decile["weighting"] == "VW")
        & (decile["decile"].astype(str).isin(labels))
    ].copy()
    if part.empty:
        return []
    part["seed"] = pd.to_numeric(part["seed"], errors="coerce")
    part["depth"] = pd.to_numeric(part["depth"], errors="coerce")
    part = part.dropna(subset=["seed", "depth"])
    part["seed"] = part["seed"].astype(int)
    part["depth"] = part["depth"].astype(int)
    part["decile_label"] = part["decile"].astype(str)
    coverage = (
        part.drop_duplicates(["family", "seed", "depth", "decile_label"])
        .groupby(["family", "seed", "depth"])["decile_label"]
        .nunique()
        .reset_index(name="n_deciles")
    )
    coverage = coverage[coverage["n_deciles"] >= len(labels)]
    left_cells = set(coverage[coverage["family"] == left_family][["seed", "depth"]].itertuples(index=False, name=None))
    if right_family is None:
        return sorted((int(seed), int(depth)) for seed, depth in left_cells)
    right_cells = set(coverage[coverage["family"] == right_family][["seed", "depth"]].itertuples(index=False, name=None))
    return sorted((int(seed), int(depth)) for seed, depth in left_cells & right_cells)


def _filter_decile_cells(frame: pd.DataFrame, cells: list[tuple[int, int]]) -> pd.DataFrame:
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


def _turnover_lookup(
    turnover: pd.DataFrame,
    *,
    families: tuple[str, ...],
    group: str = "ALL",
    weighting: str = "VW",
) -> dict[tuple[str, int, int], pd.Series]:
    part = turnover[
        (turnover["family"].isin(families))
        & (turnover["group"] == group)
        & (turnover["weighting"] == weighting)
    ].copy()
    if part.empty:
        return {}
    part["seed"] = pd.to_numeric(part["seed"], errors="coerce")
    part["depth"] = pd.to_numeric(part["depth"], errors="coerce")
    part["turnover"] = pd.to_numeric(part["turnover"], errors="coerce")
    part = part.dropna(subset=["seed", "depth", "turnover"])
    lookup: dict[tuple[str, int, int], pd.Series] = {}
    for (family, seed, depth), block in part.groupby(["family", "seed", "depth"], sort=False):
        lookup[(str(family), int(seed), int(depth))] = block.sort_values("DATE").set_index("DATE")["turnover"]
    return lookup


def _net_cost_panel(ret: pd.Series, turnover: pd.Series) -> pd.DataFrame:
    if ret.empty:
        return pd.DataFrame(columns=["ret", "turnover"])
    clean_ret = pd.to_numeric(ret, errors="coerce").dropna().sort_index()
    if clean_ret.empty:
        return pd.DataFrame(columns=["ret", "turnover"])
    clean_turnover = pd.to_numeric(turnover, errors="coerce").dropna().copy()
    clean_turnover.loc[clean_ret.index[0]] = INITIAL_LONG_SHORT_GROSS_TURNOVER
    return pd.concat(
        [clean_ret.rename("ret"), clean_turnover.sort_index().rename("turnover")],
        axis=1,
        join="inner",
    ).dropna()


def _net_cost_return_series(ret: pd.Series, turnover: pd.Series, cost_bps: float) -> pd.Series:
    joined = _net_cost_panel(ret, turnover)
    if joined.empty:
        return pd.Series(dtype=float)
    return (joined["ret"] - joined["turnover"] * float(cost_bps) / 10000.0).sort_index()


def _net_cost_stats(ret: pd.Series, turnover: pd.Series, cost_bps: float) -> dict[str, float]:
    joined = _net_cost_panel(ret, turnover)
    if joined.empty:
        return {"mean": np.nan, "sr": np.nan, "turnover": np.nan}
    net = joined["ret"] - joined["turnover"] * float(cost_bps) / 10000.0
    if net.empty:
        return {"mean": np.nan, "sr": np.nan, "turnover": np.nan}
    vol = net.std(ddof=1)
    return {
        "mean": float(net.mean() * 100.0),
        "sr": float(net.mean() / vol * np.sqrt(12.0)) if vol else np.nan,
        "turnover": float(joined["turnover"].mean() * 100.0),
    }


def _nber_business_cycle_intervals(dates: pd.Index | pd.Series) -> list[dict[str, object]]:
    periods = pd.PeriodIndex(pd.to_datetime(pd.Index(dates).dropna()), freq="M")
    if periods.empty:
        return []
    sample_start = periods.min()
    sample_end = periods.max()
    intervals: list[dict[str, object]] = []
    cursor = sample_start
    for peak_text, trough_text in NBER_BUSINESS_CYCLE_PEAK_TROUGH:
        peak = pd.Period(peak_text, freq="M")
        trough = pd.Period(trough_text, freq="M")
        recession_start = peak + 1
        recession_end = trough

        expansion_start = cursor
        expansion_end = min(peak, sample_end)
        if expansion_start <= expansion_end and expansion_end >= sample_start:
            start = max(expansion_start, sample_start)
            intervals.append(
                {
                    "state": "Expansion",
                    "start": start,
                    "end": expansion_end,
                    "period": f"{start.strftime('%Y:%m')}--{expansion_end.strftime('%Y:%m')}",
                }
            )

        start = max(recession_start, sample_start)
        end = min(recession_end, sample_end)
        if start <= end:
            intervals.append(
                {
                    "state": "Recession",
                    "start": start,
                    "end": end,
                    "period": f"{start.strftime('%Y:%m')}--{end.strftime('%Y:%m')}",
                }
            )
        cursor = trough + 1

    if cursor <= sample_end:
        intervals.append(
            {
                "state": "Expansion",
                "start": cursor,
                "end": sample_end,
                "period": f"{cursor.strftime('%Y:%m')}--{sample_end.strftime('%Y:%m')}",
            }
        )
    return intervals


def _interval_gap_stats(
    residual_return: pd.Series,
    control_return: pd.Series,
    *,
    start: pd.Period,
    end: pd.Period,
) -> dict[str, float]:
    joined = pd.concat(
        [residual_return.rename("res"), control_return.rename("base")],
        axis=1,
        join="inner",
    ).dropna()
    if joined.empty:
        return {
            "months": 0,
            "residual_mean": np.nan,
            "benchmark_mean": np.nan,
            "mean_gap": np.nan,
            "mean_se": np.nan,
            "mean_t": np.nan,
        }
    joined["period"] = pd.PeriodIndex(joined.index, freq="M")
    block = joined[(joined["period"] >= start) & (joined["period"] <= end)]
    if block.empty:
        return {
            "months": 0,
            "residual_mean": np.nan,
            "benchmark_mean": np.nan,
            "mean_gap": np.nan,
            "mean_se": np.nan,
            "mean_t": np.nan,
        }
    gap = block["res"] - block["base"]
    months = int(len(block))
    enough_months = months >= MARKET_STATE_INFERENCE_MIN_MONTHS
    mean_se = _newey_west_mean_se(gap, lags=12) if enough_months else np.nan
    return {
        "months": months,
        "residual_mean": float(block["res"].mean() * 100.0),
        "benchmark_mean": float(block["base"].mean() * 100.0),
        "mean_gap": float(gap.mean() * 100.0),
        "mean_se": float(mean_se * 100.0) if np.isfinite(mean_se) else np.nan,
        "mean_t": float(gap.mean() / mean_se) if np.isfinite(mean_se) and mean_se else np.nan,
    }


def _newey_west_mean_se(values: pd.Series, *, lags: int = 12) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    n = len(x)
    if n <= 2:
        return np.nan
    demeaned = x - x.mean()
    max_lag = min(int(lags), n - 1)
    long_run_var = float(np.dot(demeaned, demeaned) / n)
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        gamma = float(np.dot(demeaned[lag:], demeaned[:-lag]) / n)
        long_run_var += 2.0 * weight * gamma
    se = np.sqrt(max(long_run_var, 0.0) / n)
    return float(se) if se else np.nan


def _newey_west_tstat(values: pd.Series, *, lags: int = 12) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna()
    se = _newey_west_mean_se(x, lags=lags)
    return float(x.mean() / se) if np.isfinite(se) and se else np.nan


def _monthly_decile_slope_for_cells(decile: pd.DataFrame, *, family: str, cells: list[tuple[int, int]]) -> pd.Series:
    part = decile[
        (decile["family"] == family)
        & (decile["group"] == "ALL")
        & (decile["weighting"] == "VW")
        & (decile["decile"].isin([str(i) for i in range(1, 11)]))
    ].copy()
    part = _filter_decile_cells(part, cells)
    if part.empty:
        return pd.Series(dtype=float)
    part["rank"] = pd.to_numeric(part["decile"], errors="coerce")
    part["return"] = pd.to_numeric(part["return"], errors="coerce")
    slopes = []
    x = np.arange(1.0, 11.0)
    x = x - x.mean()
    denom = float(np.dot(x, x))
    for date, date_block in part.groupby("DATE"):
        monthly = (
            date_block.dropna(subset=["rank", "return"])
            .groupby("rank")["return"]
            .mean()
            .reindex(range(1, 11))
        )
        y = monthly.to_numpy(dtype=float)
        if np.isnan(y).any():
            continue
        y = y - y.mean()
        slopes.append((date, float(np.dot(x, y) / denom)))
    if not slopes:
        return pd.Series(dtype=float)
    return pd.Series({date: slope for date, slope in slopes}).sort_index()


def _conditional_forecast_content_summary() -> pd.DataFrame:
    path = SOURCE_DIR / "stock_level_conditional_forecast_content_monthly.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}.")
    source = pd.read_csv(path)
    required = {
        "Comparison",
        "Width Schedule",
        "Depth",
        "DATE",
        "ResNet+ Coefficient",
        "NN+ Coefficient",
    }
    missing = sorted(required.difference(source.columns))
    if missing:
        raise ValueError(
            "stock_level_conditional_forecast_content_monthly.csv is missing required columns: "
            + ", ".join(missing)
        )

    frame = source.copy()
    for column in ("Width Schedule", "Depth", "ResNet+ Coefficient", "NN+ Coefficient"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["DATE"] = pd.to_datetime(frame["DATE"], errors="coerce")
    if frame[list(required)].isna().any().any():
        raise ValueError("Conditional forecast content source contains missing or invalid required values.")
    if not frame["Comparison"].eq("ResNet+ vs NN+").all():
        raise ValueError("Conditional forecast content source must contain only the ResNet+ versus NN+ comparison.")
    if frame.duplicated(["Width Schedule", "Depth", "DATE"]).any():
        raise ValueError("Conditional forecast content source contains duplicate cell month rows.")
    frame = frame.sort_values(["Width Schedule", "Depth", "DATE"]).reset_index(drop=True)

    cell_dates = frame.groupby(["Width Schedule", "Depth"])["DATE"].agg(tuple)
    reference_dates = cell_dates.iloc[0]
    if (
        len(reference_dates) != 443
        or reference_dates[0] != pd.Timestamp("1987-01-31")
        or reference_dates[-1] != pd.Timestamp("2023-11-30")
        or not cell_dates.map(lambda dates: dates == reference_dates).all()
    ):
        raise ValueError(
            "Every conditional forecast cell must cover the same 443 months from "
            "January 1987 through November 2023."
        )

    expected_specs = {"Shallow": 10, "Medium": 20, "Deep": 40}
    rows: list[dict[str, object]] = []
    for depth_group, lower_depth, upper_depth in DEPTH_BANDS:
        block = frame[frame["Depth"].between(lower_depth, upper_depth)].copy()
        specs = block[["Width Schedule", "Depth"]].drop_duplicates().shape[0]
        if specs != expected_specs[depth_group]:
            raise ValueError(
                f"Expected {expected_specs[depth_group]} {depth_group.lower()} conditional forecast cells, "
                f"found {specs}."
            )
        monthly_counts = block.groupby("DATE").size()
        if len(monthly_counts) != 443 or not monthly_counts.eq(specs).all():
            raise ValueError(
                f"Every month must include all {specs} {depth_group.lower()} conditional forecast cells."
            )
        monthly = block.groupby("DATE")[["ResNet+ Coefficient", "NN+ Coefficient"]].mean()
        residual = monthly["ResNet+ Coefficient"]
        benchmark = monthly["NN+ Coefficient"]
        difference = residual - benchmark
        rows.append(
            {
                "Depth": depth_group,
                "Specs": int(specs),
                "ResNet+": float(residual.mean() * 100.0),
                "t(ResNet+)": _newey_west_tstat(residual),
                "NN+": float(benchmark.mean() * 100.0),
                "t(NN+)": _newey_west_tstat(benchmark),
                "Difference": float(difference.mean() * 100.0),
                "t(Difference)": _newey_west_tstat(difference),
                "Months": int(len(monthly)),
            }
        )
    return pd.DataFrame(rows)


def analyze_predictor_list_table() -> dict | None:
    if not PREDICTOR_LIST_SOURCE.exists():
        return
    required = ["order", "acronym", "description", "block", "source", "frequency"]
    df = pd.read_csv(PREDICTOR_LIST_SOURCE)
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing predictor-list source columns: {missing}")
    table = df[required].copy()
    table["order"] = pd.to_numeric(table["order"], errors="raise").astype(int)
    table = table.sort_values("order").reset_index(drop=True)
    if len(table) != 161:
        raise ValueError(f"Expected 161 predictors, found {len(table)} in {PREDICTOR_LIST_SOURCE}.")
    duplicated = table.loc[table["acronym"].duplicated(), "acronym"].tolist()
    if duplicated:
        raise ValueError(f"Duplicate predictors in {PREDICTOR_LIST_SOURCE}: {duplicated}")
    return {'table': table}


def analyze_final_baseline_performance_table() -> dict | None:
    brief = completed_results(REPORT_FAMILIES, include_fallback=True)
    brief = filter_noncollapsed_rows(brief).dropna(subset=[VW_SR]).copy()
    if brief.empty:
        return
    forecast_r2 = _load_forecast_level_r2()
    brief = brief.merge(forecast_r2, on=["family", "seed", "depth"], how="left")
    decile = load_decile_returns(include_fallback=True)
    decile_stats = _vw_ls_cell_stats(decile)
    brief = brief.merge(decile_stats, on=["family", "seed", "depth"], how="left")
    brief["VW SR table"] = brief["VW SR from returns"].fillna(brief[VW_SR])
    ls_returns = _ls_return_lookup(decile, families=REPORT_FAMILIES)
    turnovers = _turnover_lookup(
        load_turnover_panel(include_fallback=True), families=REPORT_FAMILIES
    )
    ff5_factors = _load_ff5_factors()

    delta_sr_by_key: dict[tuple[str, str], tuple[float, str]] = {}
    regular_decile = filter_noncollapsed_rows(decile)
    for residual_model, benchmark_model in (("ResNet", "NN"), ("ResNet+", "NN+")):
        pairs = _paired_family_pair(residual_model, benchmark_model).dropna(
            subset=[
                f"{residual_model} {VW_SR}",
                f"{benchmark_model} {VW_SR}",
            ]
        )
        for label, lo, hi in BASELINE_DEPTH_BANDS:
            band_pairs = pairs[
                (pairs["depth"].astype(int) >= lo)
                & (pairs["depth"].astype(int) <= hi)
            ]
            specs = _paired_ls_specs(
                regular_decile,
                left_family=residual_model,
                right_family=benchmark_model,
                pairs=band_pairs,
            )
            test = (
                _sr_ledoit_wolf_test(
                    specs,
                    seed=_sr_lw_seed(residual_model, benchmark_model, label),
                )
                if specs
                else {}
            )
            if test:
                delta_sr_by_key[(residual_model, label)] = (
                    float(test["delta"]),
                    _sr_stars(float(test["p_value"])),
                )

    rows = []
    for model in REPORT_MODEL_ORDER:
        part = brief[brief["family"] == model].dropna(subset=["VW SR table"]).copy()
        if part.empty:
            continue
        for label, lo, hi in BASELINE_DEPTH_BANDS:
            band = part[(part["depth"].astype(int) >= lo) & (part["depth"].astype(int) <= hi)]
            if band.empty:
                continue
            row = _baseline_mean_row(band, depth_group=label)
            cells = [
                (int(seed), int(depth))
                for seed, depth in band[["seed", "depth"]].itertuples(index=False, name=None)
            ]
            row["Turnover"] = _cell_weighted_mean_from_values([
                float(_net_cost_panel(
                    ls_returns[(model, seed, depth)],
                    turnovers[(model, seed, depth)],
                )["turnover"].mean() * 100.0)
                for seed, depth in cells
            ])
            group_return = _cell_averaged_return_from_lookup(
                ls_returns,
                family=model,
                cells=cells,
            )
            group_alpha, group_alpha_t = _ff5_alpha_for_return(group_return, ff5_factors)
            row["FF5 $\\alpha$"] = group_alpha
            row["$t$(FF5)"] = group_alpha_t
            delta_sr, delta_sr_stars = delta_sr_by_key.get((model, label), (np.nan, ""))
            row["$\\Delta$ VW SR"] = delta_sr
            row["Delta VW SR stars"] = delta_sr_stars
            rows.append(row)
    table = pd.DataFrame(rows).rename(columns={"Depth Group": "Depth"})

    return {'table': table}


def analyze_depth_group_sharpe_contrasts() -> dict | None:
    brief = filter_noncollapsed_rows(
        completed_results(REPORT_FAMILIES, include_fallback=True)
    ).dropna(subset=[VW_SR])
    if brief.empty:
        return
    decile = filter_noncollapsed_rows(load_decile_returns(include_fallback=True))
    returns = _ls_return_lookup(decile, families=REPORT_FAMILIES)
    rows: list[dict[str, object]] = []
    for model in REPORT_MODEL_ORDER:
        part = brief[brief["family"] == model].copy()
        shallow_cells = sorted(
            (int(seed), int(depth))
            for seed, depth in part.loc[
                part["depth"].astype(int).between(1, 5), ["seed", "depth"]
            ].itertuples(index=False, name=None)
        )
        deep_cells = sorted(
            (int(seed), int(depth))
            for seed, depth in part.loc[
                part["depth"].astype(int).between(11, 20), ["seed", "depth"]
            ].itertuples(index=False, name=None)
        )
        series = {
            (int(seed), int(depth)): returns[(model, int(seed), int(depth))]
            for seed, depth in shallow_cells + deep_cells
        }
        return_panel = pd.concat(series, axis=1, join="inner").sort_index()
        test = _weighted_group_sr_contrast_test(
            return_panel,
            shallow_columns=shallow_cells,
            deep_columns=deep_cells,
            seed=DEPTH_CONTRAST_BOOTSTRAP_SEEDS[model],
        )
        if not test:
            continue
        expected_shallow = _cell_weighted_mean(
            part[part["depth"].astype(int).between(1, 5)], VW_SR
        )
        expected_deep = _cell_weighted_mean(
            part[part["depth"].astype(int).between(11, 20)], VW_SR
        )
        if not np.isclose(test["shallow_sr"], expected_shallow, atol=1e-12):
            raise ValueError(f"{model}: shallow SR does not reconcile to Table 2.")
        if not np.isclose(test["deep_sr"], expected_deep, atol=1e-12):
            raise ValueError(f"{model}: deep SR does not reconcile to Table 2.")
        rows.append(
            {
                "Model": model,
                "Deep - Shallow SR": Estimate(
                    float(test["delta"]), _sr_stars(float(test["p_value"]))
                ),
                "$t$": float(test["t"]),
                "$p$": float(test["p_value"]),
                "CI Low": float(test["ci_low"]),
                "CI High": float(test["ci_high"]),
                "Shallow SR": float(test["shallow_sr"]),
                "Deep SR": float(test["deep_sr"]),
                "Shallow Cells": int(test["shallow_cells"]),
                "Deep Cells": int(test["deep_cells"]),
                "Months": int(test["months"]),
                "Bootstrap Draws": int(test["bootstrap_draws"]),
            }
        )
    if len(rows) != len(REPORT_MODEL_ORDER):
        raise ValueError(f"Expected {len(REPORT_MODEL_ORDER)} depth contrasts, found {len(rows)}.")
    return {"table": pd.DataFrame(rows)}


def analyze_full_decile_monotonicity_table() -> dict | None:
    decile = filter_noncollapsed_rows(load_decile_returns(include_fallback=True))
    if not DECILE_PORTFOLIO_STATISTICS_SOURCE.exists():
        raise FileNotFoundError(
            f"Missing {DECILE_PORTFOLIO_STATISTICS_SOURCE}. "
            "Run generator.collect_results.collect_decile_portfolio_statistics() first."
        )
    cell_stats = pd.read_csv(DECILE_PORTFOLIO_STATISTICS_SOURCE)
    cell_stats = cell_stats[
        (cell_stats["group"] == "ALL")
        & cell_stats["family"].isin(("ResNet", "NN", "ResNet+", "NN+"))
    ].copy()
    cell_stats = filter_noncollapsed_rows(cell_stats)
    cell_stats["rank"] = cell_stats["rank"].map(normalize_decile_label)
    for column in ("seed", "depth", "pred", "avg", "sd", "sr"):
        cell_stats[column] = pd.to_numeric(cell_stats[column], errors="coerce")
    cell_stats = cell_stats.dropna(subset=["seed", "depth", "rank", "pred", "avg", "sd", "sr"])
    cell_stats["seed"] = cell_stats["seed"].astype(int)
    cell_stats["depth"] = cell_stats["depth"].astype(int)
    if cell_stats.empty:
        return
    cell_stats = cell_stats.rename(
        columns={"pred": "Pred", "avg": "Avg", "sd": "SD", "sr": "SR"}
    )

    cells_by_family = {
        family: _completed_decile_cells(decile, left_family=family)
        for family in ("ResNet", "NN", "ResNet+", "NN+")
    }

    depth_groups = DEPTH_BANDS
    performance: dict[tuple[str, str], pd.DataFrame] = {}
    for family in ("ResNet", "NN", "ResNet+", "NN+"):
        family_stats = cell_stats[cell_stats["family"] == family].copy()
        for depth_label, lo, hi in depth_groups:
            cells = [cell for cell in cells_by_family[family] if lo <= cell[1] <= hi]
            cell_index = pd.MultiIndex.from_tuples(cells, names=["seed", "depth"])
            row_index = pd.MultiIndex.from_frame(family_stats[["seed", "depth"]])
            block = family_stats[row_index.isin(cell_index)]
            performance[(family, depth_label)] = block.groupby("rank")[["Pred", "Avg", "SD", "SR"]].mean()

    return {'performance': performance, 'depth_groups': depth_groups}


def analyze_transaction_cost_table() -> dict | None:
    decile = filter_noncollapsed_rows(load_decile_returns(include_fallback=True))
    turnover = filter_noncollapsed_rows(load_turnover_panel(include_fallback=True))
    baseline_brief = filter_noncollapsed_rows(
        completed_results(REPORT_FAMILIES, include_fallback=True)
    ).dropna(subset=[VW_SR])
    comparison_specs = (('ResNet vs NN', 'ResNet', 'NN'), ('ResNet+ vs NN+', 'ResNet+', 'NN+'))
    costs = (0.0, 10.0, 50.0)
    expected_months = pd.period_range('1987-02', '2023-12', freq='M')
    rows = []
    for comparison, residual_family, control_family in comparison_specs:
        families = (residual_family, control_family)
        returns = _ls_return_lookup(decile, families=families)
        turnovers = _turnover_lookup(turnover, families=families)
        residual_cells = {(seed, depth) for family, seed, depth in returns if family == residual_family and (family, seed, depth) in turnovers}
        control_cells = {(seed, depth) for family, seed, depth in returns if family == control_family and (family, seed, depth) in turnovers}
        paired_cells = sorted(residual_cells & control_cells)
        if not paired_cells:
            continue
        net_returns_by_cost: dict[str, dict[tuple[str, int, int], pd.Series]] = {str(int(cost)): {} for cost in costs}
        cell_rows = []
        for seed, depth in paired_cells:
            row: dict[str, float | int] = {'seed': int(seed), 'depth': int(depth)}
            residual_key = (residual_family, int(seed), int(depth))
            control_key = (control_family, int(seed), int(depth))
            for key in (residual_key, control_key):
                joined = _net_cost_panel(returns[key], turnovers[key])
                months = pd.DatetimeIndex(joined.index).to_period('M')
                if not months.equals(expected_months):
                    raise ValueError(f'{key}: SR and turnover must share all 443 return months.')
            for cost in costs:
                residual_stats = _net_cost_stats(returns[residual_key], turnovers[residual_key], cost)
                control_stats = _net_cost_stats(returns[control_key], turnovers[control_key], cost)
                suffix = str(int(cost))
                net_returns_by_cost[suffix][residual_key] = _net_cost_return_series(returns[residual_key], turnovers[residual_key], cost)
                net_returns_by_cost[suffix][control_key] = _net_cost_return_series(returns[control_key], turnovers[control_key], cost)
                row[f'residual_sr_{suffix}'] = residual_stats['sr']
                row[f'control_sr_{suffix}'] = control_stats['sr']
                row[f'residual_turnover_{suffix}'] = residual_stats['turnover']
                row[f'control_turnover_{suffix}'] = control_stats['turnover']
                row[f'delta_sr_{suffix}'] = residual_stats['sr'] - control_stats['sr']
            cell_rows.append(row)
        cells = pd.DataFrame(cell_rows)
        # Model SRs use each family's available specifications; gaps and turnover use matched pairs.
        model_sr_rows = []
        for family in families:
            family_brief = baseline_brief[baseline_brief['family'] == family]
            for seed, depth in family_brief[['seed', 'depth']].itertuples(index=False, name=None):
                key = (family, int(seed), int(depth))
                if key not in returns or key not in turnovers:
                    raise ValueError(f'{key}: missing returns or turnover for a baseline SR specification.')
                joined = _net_cost_panel(returns[key], turnovers[key])
                if not pd.DatetimeIndex(joined.index).to_period('M').equals(expected_months):
                    raise ValueError(f'{key}: SR and turnover must share all 443 return months.')
                model_row = {'family': family, 'seed': int(seed), 'depth': int(depth)}
                for cost in costs:
                    model_row[f'sr_{int(cost)}'] = _net_cost_stats(returns[key], turnovers[key], cost)['sr']
                model_sr_rows.append(model_row)
        model_sr_cells = pd.DataFrame(model_sr_rows)
        for depth_group, lo, hi in DEPTH_BANDS:
            block = cells[(cells['depth'] >= lo) & (cells['depth'] <= hi)]
            if block.empty:
                continue
            model_sr_block = model_sr_cells[model_sr_cells['depth'].between(lo, hi)]
            row = {'Comparison': comparison, 'Depth Group': depth_group}
            for cost in costs:
                suffix = str(int(cost))
                specs = _paired_return_specs_from_lookup(net_returns_by_cost[suffix], left_family=residual_family, right_family=control_family, pairs=block)
                lw_test = _sr_ledoit_wolf_test(specs, seed=_sr_lw_seed(residual_family, control_family, depth_group, offset=100 * int(cost))) if specs else {}
                delta_value = _cell_weighted_mean(block, f'delta_sr_{suffix}')
                row[f'$\\Delta$ SR, {suffix} bps'] = Estimate(delta_value, _sr_stars(float(lw_test['p_value'])) if lw_test else '')
                row[f'Residual SR, {suffix} bps'] = _cell_weighted_mean(model_sr_block[model_sr_block['family'] == residual_family], f'sr_{suffix}')
                row[f'Benchmark SR, {suffix} bps'] = _cell_weighted_mean(model_sr_block[model_sr_block['family'] == control_family], f'sr_{suffix}')
                row[f'Residual Turnover, {suffix} bps'] = _cell_weighted_mean(block, f'residual_turnover_{suffix}')
                row[f'Benchmark Turnover, {suffix} bps'] = _cell_weighted_mean(block, f'control_turnover_{suffix}')
                row[f'SR p-value, {suffix} bps'] = float(lw_test['p_value']) if lw_test else np.nan
            row['Paired specifications'] = len(block)
            row['Months'] = 443
            rows.append(row)
    table = pd.DataFrame(rows).rename(columns={'Depth Group': 'Depth'})
    if table.empty:
        return
    cost_labels = [str(int(cost)) for cost in costs]
    return {'table': table, 'comparison_specs': comparison_specs, 'cost_labels': cost_labels}


def analyze_size_exclusion_summary_table() -> dict | None:
    panel = filter_noncollapsed_rows(load_brief_panel(include_fallback=True))
    decile = filter_noncollapsed_rows(load_decile_returns(include_fallback=True))
    group_specs = (
        ("All Stocks", "ALL"),
        ("Exclude Smallest 20\\%", "TOP80%"),
        ("Exclude Largest 20\\%", "BOTTOM80%"),
    )
    comparison_specs = (
        ("ResNet vs NN", "ResNet", "NN"),
        ("ResNet+ vs NN+", "ResNet+", "NN+"),
    )
    rows = []
    for comparison, residual_family, control_family in comparison_specs:
        for universe, group in group_specs:
            part = panel[
                (panel["group"] == group)
                & (panel["family"].isin([residual_family, control_family]))
            ].copy()
            wide = part.pivot_table(
                index=["seed", "depth"],
                columns="family",
                values=VW_SR,
                aggfunc="first",
            )
            if residual_family not in wide.columns or control_family not in wide.columns:
                continue
            wide = wide.dropna(subset=[residual_family, control_family]).reset_index()
            if wide.empty:
                continue
            wide["seed"] = wide["seed"].astype(int)
            wide["depth"] = wide["depth"].astype(int)
            wide["delta_sr"] = wide[residual_family] - wide[control_family]
            for depth_group, lo, hi in DEPTH_BANDS:
                block = wide[(wide["depth"] >= lo) & (wide["depth"] <= hi)]
                if block.empty:
                    continue
                specs = _paired_ls_specs(
                    decile,
                    left_family=residual_family,
                    right_family=control_family,
                    pairs=block,
                    group=group,
                )
                lw_test = _sr_ledoit_wolf_test(
                    specs,
                    seed=_sr_lw_seed(residual_family, control_family, depth_group, offset=1000 * (group_specs.index((universe, group)))),
                ) if specs else {}
                rows.append(
                    {
                        "Comparison": comparison,
                        "Universe": universe,
                        "Depth Group": depth_group,
                        "$\\Delta$ SR": _cell_weighted_mean(block, "delta_sr"),
                        LW_T_COL: _safe_tstat_from_se(lw_test["delta"], lw_test["se"]) if lw_test else np.nan,
                        "SR stars": _sr_stars(float(lw_test["p_value"])) if lw_test else "",
                        "Res SR": _cell_weighted_mean(block, residual_family),
                        "Base SR": _cell_weighted_mean(block, control_family),
                    }
                )
    table = pd.DataFrame(rows)
    if table.empty:
        return

    return {'table': table, 'group_specs': group_specs, 'comparison_specs': comparison_specs}


def analyze_equal_weighted_robustness_table() -> dict | None:
    panel = filter_noncollapsed_rows(load_brief_panel(include_fallback=True))
    decile = filter_noncollapsed_rows(load_decile_returns(include_fallback=True))
    forecast_r2 = _load_forecast_level_r2()
    ff5_factors = _load_ff5_factors()
    metric = EW_SR
    comparison_specs = (('ResNet vs NN', 'ResNet', 'NN'), ('ResNet+ vs NN+', 'ResNet+', 'NN+'))
    rows = []
    for comparison, residual_family, control_family in comparison_specs:
        ls_returns = _ls_return_lookup(decile, families=(residual_family, control_family), weighting='EW')
        part = panel[(panel['group'] == 'ALL') & panel['family'].isin([residual_family, control_family])].copy()
        wide = part.pivot_table(index=['seed', 'depth'], columns='family', values=metric, aggfunc='first')
        if residual_family not in wide.columns or control_family not in wide.columns:
            continue
        wide = wide.dropna(subset=[residual_family, control_family]).reset_index()
        if wide.empty:
            continue
        wide['seed'] = wide['seed'].astype(int)
        wide['depth'] = wide['depth'].astype(int)
        wide['delta_sr'] = wide[residual_family] - wide[control_family]
        wide = _attach_forecast_level_r2(wide, left_family=residual_family, right_family=control_family, source=forecast_r2)
        residual_r2 = f'{residual_family} {TABLE_R2_OOS}'
        control_r2 = f'{control_family} {TABLE_R2_OOS}'
        wide['delta_r2_oos'] = wide[residual_r2] - wide[control_r2]
        turnover_wide = part.pivot_table(index=['seed', 'depth'], columns='family', values=EW_TURNOVER, aggfunc='first')
        if residual_family in turnover_wide.columns and control_family in turnover_wide.columns:
            turnover_wide = turnover_wide.reset_index()
            turnover_wide['seed'] = turnover_wide['seed'].astype(int)
            turnover_wide['depth'] = turnover_wide['depth'].astype(int)
            turnover_wide['delta_turnover'] = turnover_wide[residual_family] - turnover_wide[control_family]
            wide = wide.merge(turnover_wide[['seed', 'depth', 'delta_turnover']], on=['seed', 'depth'], how='left')
        else:
            wide['delta_turnover'] = np.nan
        for depth_group, lo, hi in DEPTH_BANDS:
            block = wide[(wide['depth'] >= lo) & (wide['depth'] <= hi)]
            if block.empty:
                continue
            delta_return = _cell_averaged_delta_ls_return(ls_returns, block, left_family=residual_family, right_family=control_family)
            delta_alpha, delta_alpha_t = _ff5_alpha_for_delta_return(delta_return, ff5_factors)
            specs = _paired_ls_specs(decile, left_family=residual_family, right_family=control_family, pairs=block, weighting='EW')
            lw_test = _sr_ledoit_wolf_test(specs, seed=_sr_lw_seed(residual_family, control_family, depth_group, offset=5000)) if specs else {}
            rows.append({'Comparison': comparison, 'Depth Group': depth_group, 'Res EW SR': _cell_weighted_mean(block, residual_family), 'Base EW SR': _cell_weighted_mean(block, control_family), '$\\Delta$ EW SR': Estimate(_cell_weighted_mean(block, 'delta_sr'), _sr_stars(float(lw_test['p_value'])) if lw_test else ''), LW_T_COL: _safe_tstat_from_se(lw_test['delta'], lw_test['se']) if lw_test else np.nan, '$\\Delta R^2_{OOS}$': _cell_weighted_mean(block, 'delta_r2_oos'), '$\\Delta \\alpha_{\\mathrm{FF5}}$': delta_alpha, '$t(\\Delta \\alpha)$': delta_alpha_t, '$\\Delta$ Turnover': _cell_weighted_mean(block, 'delta_turnover')})
    table = pd.DataFrame(rows).rename(columns={'Depth Group': 'Depth'})
    if table.empty:
        return
    table = table[['Comparison', 'Depth', 'Res EW SR', 'Base EW SR', '$\\Delta$ EW SR', LW_T_COL, '$\\Delta R^2_{OOS}$', '$\\Delta \\alpha_{\\mathrm{FF5}}$', '$\\Delta$ Turnover']].rename(columns={'Res EW SR': 'Res. SR', 'Base EW SR': 'Bench. SR', '$\\Delta$ EW SR': '$\\Delta$ SR', LW_T_COL: '$t$', '$\\Delta R^2_{OOS}$': '$\\Delta R^2_{\\mathrm{OOS}}$', '$\\Delta \\alpha_{\\mathrm{FF5}}$': '$\\Delta$ FF5 $\\alpha$', '$\\Delta$ Turnover': '$\\Delta$ Turn.'})
    return {'table': table}


def analyze_market_state_robustness_table() -> dict | None:
    decile = filter_noncollapsed_rows(load_decile_returns(include_fallback=True))
    comparison_specs = (
        ("ResNet vs NN", "ResNet", "NN"),
        ("ResNet+ vs NN+", "ResNet+", "NN+"),
    )
    comparison_series: dict[str, tuple[pd.Series, pd.Series]] = {}
    all_dates: list[pd.Timestamp] = []
    for comparison, residual_family, control_family in comparison_specs:
        cells = _completed_decile_cells(
            decile,
            left_family=residual_family,
            right_family=control_family,
            deciles=("LS",),
        )
        if not cells:
            continue
        ls_returns = _ls_return_lookup(decile, families=(residual_family, control_family))
        residual_return = _cell_averaged_return_from_lookup(
            ls_returns,
            family=residual_family,
            cells=cells,
        )
        control_return = _cell_averaged_return_from_lookup(
            ls_returns,
            family=control_family,
            cells=cells,
        )
        comparison_series[comparison] = (residual_return, control_return)
        all_dates.extend(pd.to_datetime(residual_return.index).tolist())
        all_dates.extend(pd.to_datetime(control_return.index).tolist())

    intervals = _nber_business_cycle_intervals(pd.Index(all_dates))
    if not intervals or not comparison_series:
        return

    rows = []
    for interval in intervals:
        row = {
            "Period": str(interval["period"]),
            "State": str(interval["state"]),
            "Months": 0,
        }
        for comparison, _, _ in comparison_specs:
            if comparison not in comparison_series:
                continue
            residual_return, control_return = comparison_series[comparison]
            stats = _interval_gap_stats(
                residual_return,
                control_return,
                start=interval["start"],
                end=interval["end"],
            )
            row["Months"] = max(int(row["Months"]), int(stats["months"]))
            prefix = "RN-NN" if comparison == "ResNet vs NN" else "RNP-NNP"
            row[f"{prefix} Residual Mean"] = stats["residual_mean"]
            row[f"{prefix} Benchmark Mean"] = stats["benchmark_mean"]
            row[f"{prefix} Mean"] = stats["mean_gap"]
            row[f"{prefix} t-stat"] = stats["mean_t"]
            row[f"{prefix} Mean Stars"] = _tstat_stars(stats["mean_t"])
        rows.append(row)

    return {'rows': rows, 'comparison_specs': comparison_specs, 'comparison_series': comparison_series}


def analyze_paired_specification_depth_profile() -> dict:
    return {"table": _raw_result_table(('ResNet+', 'NN+'))}


def analyze_resnet_vs_nn_depth_profile() -> dict:
    return {"table": _raw_result_table(('ResNet', 'NN'))}
