from __future__ import annotations

import argparse
import gc
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils.paper_source.paths import REPO_ROOT, SOURCE_DIR
else:
    from utils.paper_source.paths import REPO_ROOT, SOURCE_DIR


PAIR_COMPARISONS = [
    ("ResNet vs NN", "ResNet", "resnets", "NN", "nns"),
    ("Paired specification design", "ResNet+", "resnetps", "NN+", "nnps"),
]

DEPTH_BANDS = (
    ("Shallow", 1, 5),
    ("Medium", 6, 10),
    ("Deep", 11, 20),
    ("All", 1, 20),
)

CHARACTERISTICS = [
    ("Log market equity", "__log_mkt_cap__", "Size"),
    ("Book-to-market", "be_me", "Value"),
    ("12-month momentum", "ret_12_1", "Momentum"),
    ("1-month reversal", "ret_1_0", "Reversal"),
    ("Gross profitability", "gp_at", "Profitability"),
    ("Operating profitability", "op_at", "Profitability"),
    ("Asset growth", "at_gr1", "Investment"),
    ("Inventory change", "inv_gr1a", "Inventory"),
    ("Market beta", "beta_60m", "Risk"),
    ("Idiosyncratic volatility", "ivol_capm_21d", "Risk"),
    ("Turnover", "turnover_126d", "Liquidity"),
    ("Bid-ask spread", "bidaskhl_21d", "Liquidity"),
]

CONTROL_COLUMNS = [
    "__log_mkt_cap__",
    "be_me",
    "ret_12_1",
    "ret_1_0",
    "gp_at",
    "inv_gr1a",
    "beta_60m",
    "ivol_capm_21d",
]

DEFAULT_SUPPLEMENTARY_JOBS = 8

_STOCK_WORKER_STATE: dict[str, object] = {}
def _nw_tstat(values: pd.Series, lags: int = 12) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    n = len(x)
    if n <= 2:
        return float("nan")
    demeaned = x - x.mean()
    max_lag = min(int(lags), n - 1)
    long_run_var = float(np.dot(demeaned, demeaned) / n)
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        gamma = float(np.dot(demeaned[lag:], demeaned[:-lag]) / n)
        long_run_var += 2.0 * weight * gamma
    se = math.sqrt(max(long_run_var, 0.0) / n)
    return float(x.mean() / se) if se else float("nan")


def _load_forecast(repo_root: Path, prefix: str, start_ensemble: int, max_ensemble: int) -> pd.Series:
    pieces = []
    for year in range(1987, 2024):
        date = f"{year}1231"
        forecasts = []
        for ensemble in range(start_ensemble, max_ensemble + 1):
            path = repo_root / "asset" / prefix / "predicts" / date / f"predicts_{ensemble}.pkl"
            forecasts.append(pd.read_pickle(path))
        avg = pd.concat(forecasts, axis=1).mean(axis=1)
        pieces.append(avg)
    out = pd.concat(pieces).sort_index()
    out.name = prefix
    return out


def _load_forecast_optional(
    repo_root: Path,
    prefix: str,
    start_ensemble: int,
    max_ensemble: int,
    cache: dict[str, pd.Series | None],
) -> pd.Series | None:
    if prefix not in cache:
        try:
            cache[prefix] = _load_forecast(repo_root, prefix, start_ensemble, max_ensemble)
        except FileNotFoundError:
            cache[prefix] = None
    return cache[prefix]


def _load_stock_level_inputs(repo_root: Path) -> tuple[pd.DataFrame, pd.Series]:
    char_cols = [col for _, col, _ in CHARACTERISTICS if col != "__log_mkt_cap__"]
    X = pd.read_pickle(repo_root / "tmp" / "clean_X.pkl")
    mkt = pd.read_pickle(repo_root / "tmp" / "clean_mvel1.pkl")
    y = pd.read_pickle(repo_root / "tmp" / "clean_y.pkl")
    X = X[char_cols].copy()
    X["__log_mkt_cap__"] = np.log(pd.to_numeric(mkt, errors="coerce").clip(lower=1e-12))
    return X, y


def _stock_level_base_panel(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    return pd.concat(
        [y.rename("Realized return"), X[CONTROL_COLUMNS]],
        axis=1,
        join="inner",
    ).dropna()


def _zscore(block: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = block.copy()
    for col in cols:
        values = pd.to_numeric(out[col], errors="coerce")
        std = values.std(ddof=0)
        if not std or pd.isna(std):
            out[col] = np.nan
        else:
            out[col] = (values - values.mean()) / std
    return out


def _paired_incremental_specs(source_dir: Path) -> list[dict[str, object]]:
    path = source_dir / "brief_results_all.csv"
    if not path.exists():
        return [
            {
                "comparison": comparison,
                "residual_model": residual_model,
                "residual_prefix": f"{residual_stem}{seed}d{depth}",
                "control_model": control_model,
                "control_prefix": f"{control_stem}{seed}d{depth}",
                "seed": seed,
                "depth": depth,
            }
            for comparison, residual_model, residual_stem, control_model, control_stem, seed, depth in [
                ("ResNet vs NN", "ResNet", "resnets", "NN", "nns", 5, 10),
                ("Paired specification design", "ResNet+", "resnetps", "NN+", "nnps", 3, 20),
            ]
        ]

    brief = pd.read_csv(path)
    for column in ("seed", "depth"):
        brief[column] = pd.to_numeric(brief[column], errors="coerce")
    metric = "Value Weighted Long Short Sharpe Ratio"
    if metric in brief.columns:
        brief[metric] = pd.to_numeric(brief[metric], errors="coerce")
        brief = brief.dropna(subset=[metric])
    fallback_columns = [
        "Seeded Tie-Split Decile Fallback Months",
        "Constant-Predict Decile Fallback Months",
    ]
    fallback_mask = pd.Series(False, index=brief.index)
    for column in fallback_columns:
        if column in brief.columns:
            fallback_mask |= pd.to_numeric(brief[column], errors="coerce").fillna(0.0) > 0.0
    brief = brief.loc[~fallback_mask].copy()
    specs: list[dict[str, object]] = []
    for comparison, residual_model, residual_stem, control_model, control_stem in PAIR_COMPARISONS:
        residual_cells = {
            (int(row.seed), int(row.depth))
            for row in brief[brief["family"].eq(residual_model)][["seed", "depth"]].itertuples(index=False)
            if pd.notna(row.seed) and pd.notna(row.depth)
        }
        control_cells = {
            (int(row.seed), int(row.depth))
            for row in brief[brief["family"].eq(control_model)][["seed", "depth"]].itertuples(index=False)
            if pd.notna(row.seed) and pd.notna(row.depth)
        }
        for seed, depth in sorted(residual_cells & control_cells):
            specs.append(
                {
                    "comparison": comparison,
                    "residual_model": residual_model,
                    "residual_prefix": f"{residual_stem}{seed}d{depth}",
                    "control_model": control_model,
                    "control_prefix": f"{control_stem}{seed}d{depth}",
                    "seed": seed,
                    "depth": depth,
                }
            )
    return specs


def _bounded_jobs(jobs: int, task_count: int) -> int:
    if task_count <= 0:
        return 1
    return max(1, min(int(jobs), int(task_count)))


def _monthly_incremental_coefficients(
    *,
    base_panel: pd.DataFrame,
    residual: pd.Series,
    control: pd.Series,
    residual_model: str,
    control_model: str,
) -> pd.DataFrame:
    panel = pd.concat(
        [
            base_panel,
            residual.rename(residual_model),
            control.rename(control_model),
        ],
        axis=1,
        join="inner",
    ).dropna()
    reg_cols = [residual_model, control_model] + CONTROL_COLUMNS
    coef_rows = []
    for date, block in panel.groupby(level="DATE"):
        if len(block) < len(reg_cols) + 20:
            continue
        block = _zscore(block, reg_cols).dropna()
        if len(block) < len(reg_cols) + 20:
            continue
        x = np.column_stack([np.ones(len(block)), block[reg_cols].to_numpy(dtype=float)])
        yy = block["Realized return"].to_numpy(dtype=float)
        beta = np.linalg.pinv(x.T @ x) @ (x.T @ yy)
        coef_rows.append(pd.Series(beta[1:3], index=[residual_model, control_model], name=date))
    if not coef_rows:
        return pd.DataFrame(columns=[residual_model, control_model])
    return pd.DataFrame(coef_rows).sort_index()


def _stock_level_record_for_spec(
    *,
    repo_root: Path,
    spec: dict[str, object],
    start_ensemble: int,
    max_ensemble: int,
    base_panel: pd.DataFrame,
    forecast_cache: dict[str, pd.Series | None] | None = None,
) -> dict[str, object] | None:
    forecast_cache = forecast_cache if forecast_cache is not None else {}
    pinned_prefixes = set(forecast_cache)
    residual_prefix = str(spec["residual_prefix"])
    control_prefix = str(spec["control_prefix"])
    residual = _load_forecast_optional(
        repo_root,
        residual_prefix,
        start_ensemble,
        max_ensemble,
        forecast_cache,
    )
    control = _load_forecast_optional(
        repo_root,
        control_prefix,
        start_ensemble,
        max_ensemble,
        forecast_cache,
    )
    for prefix in (residual_prefix, control_prefix):
        if prefix not in pinned_prefixes:
            forecast_cache.pop(prefix, None)
    if residual is None or control is None:
        gc.collect()
        return None
    coefs = _monthly_incremental_coefficients(
        base_panel=base_panel,
        residual=residual,
        control=control,
        residual_model=str(spec["residual_model"]),
        control_model=str(spec["control_model"]),
    )
    gc.collect()
    if coefs.empty:
        return None
    return {**spec, "coefficients": coefs}


def _init_stock_worker(repo_root: str, start_ensemble: int, max_ensemble: int) -> None:
    root = Path(repo_root).resolve()
    X, y = _load_stock_level_inputs(root)
    _STOCK_WORKER_STATE.clear()
    _STOCK_WORKER_STATE.update(
        {
            "repo_root": root,
            "start_ensemble": int(start_ensemble),
            "max_ensemble": int(max_ensemble),
            "base_panel": _stock_level_base_panel(X, y),
        }
    )


def _stock_worker(spec: dict[str, object]) -> dict[str, object] | None:
    return _stock_level_record_for_spec(
        repo_root=_STOCK_WORKER_STATE["repo_root"],
        spec=spec,
        start_ensemble=_STOCK_WORKER_STATE["start_ensemble"],
        max_ensemble=_STOCK_WORKER_STATE["max_ensemble"],
        base_panel=_STOCK_WORKER_STATE["base_panel"],
    )


def _summarize_incremental_coefficients(
    records: list[dict[str, object]],
    *,
    comparison: str,
    residual_model: str,
    control_model: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    comparison_records = [record for record in records if record["comparison"] == comparison]
    for depth_group, lo, hi in DEPTH_BANDS:
        block = [record for record in comparison_records if lo <= int(record["depth"]) <= hi]
        if not block:
            continue
        residual_monthly = pd.concat(
            [record["coefficients"][residual_model] for record in block],
            axis=1,
        ).mean(axis=1)
        control_monthly = pd.concat(
            [record["coefficients"][control_model] for record in block],
            axis=1,
        ).mean(axis=1)
        joined = pd.concat(
            [residual_monthly.rename("residual"), control_monthly.rename("control")],
            axis=1,
            join="inner",
        ).dropna()
        if joined.empty:
            continue
        delta = joined["residual"] - joined["control"]
        rows.append(
            {
                "Comparison": comparison,
                "Depth Group": depth_group,
                "Specs": int(len(block)),
                "Months": int(len(delta)),
                "Delta Coefficient/%": float(delta.mean() * 100.0),
                "t-stat Delta": _nw_tstat(delta),
                "Res Coefficient/%": float(joined["residual"].mean() * 100.0),
                "Base Coefficient/%": float(joined["control"].mean() * 100.0),
            }
        )
    return rows


def compute_stock_level_incremental_regression(
    repo_root: Path,
    source_dir: Path,
    *,
    start_ensemble: int,
    max_ensemble: int,
    X: pd.DataFrame | None = None,
    y: pd.Series | None = None,
    forecast_cache: dict[str, pd.Series | None] | None = None,
    jobs: int = DEFAULT_SUPPLEMENTARY_JOBS,
    progress: bool = False,
) -> pd.DataFrame:
    specs = _paired_incremental_specs(source_dir)
    coefficient_records: list[dict[str, object]] = []

    if jobs > 1 and X is None and y is None and not forecast_cache:
        workers = _bounded_jobs(jobs, len(specs))
        if progress:
            print(f"[stock-level] running {len(specs)} cells with {workers} jobs", flush=True)
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_stock_worker,
            initargs=(str(repo_root), int(start_ensemble), int(max_ensemble)),
        ) as executor:
            future_to_spec = {executor.submit(_stock_worker, spec): spec for spec in specs}
            completed = 0
            for future in as_completed(future_to_spec):
                completed += 1
                spec = future_to_spec[future]
                record = future.result()
                if record is not None:
                    coefficient_records.append(record)
                if progress:
                    print(
                        "[stock-level] "
                        f"{completed}/{len(specs)} {spec['comparison']} "
                        f"s{spec['seed']}d{spec['depth']}",
                        flush=True,
                    )
    else:
        if X is None or y is None:
            X, y = _load_stock_level_inputs(repo_root)
        base_panel = _stock_level_base_panel(X, y)
        forecast_cache = forecast_cache or {}
        for idx, spec in enumerate(specs, start=1):
            if progress:
                print(
                    "[stock-level] "
                    f"{idx}/{len(specs)} {spec['comparison']} "
                    f"s{spec['seed']}d{spec['depth']}",
                    flush=True,
                )
            record = _stock_level_record_for_spec(
                repo_root=repo_root,
                spec=spec,
                start_ensemble=start_ensemble,
                max_ensemble=max_ensemble,
                base_panel=base_panel,
                forecast_cache=forecast_cache,
            )
            if record is not None:
                coefficient_records.append(record)

    reg_rows: list[dict[str, object]] = []
    for comparison, residual_model, _, control_model, _ in PAIR_COMPARISONS:
        reg_rows.extend(
            _summarize_incremental_coefficients(
                coefficient_records,
                comparison=comparison,
                residual_model=residual_model,
                control_model=control_model,
            )
        )
    out = pd.DataFrame(reg_rows)
    out.to_csv(source_dir / "stock_level_incremental_regression.csv", index=False)
    if progress:
        print(f"[stock-level] wrote {len(out)} summary rows", flush=True)
    return out


def run_supplementary_experiments(
    *,
    repo_root: str | Path = REPO_ROOT,
    output_root: str | Path = SOURCE_DIR,
    start_ensemble: int = 0,
    max_ensemble: int = 9,
    jobs: int = DEFAULT_SUPPLEMENTARY_JOBS,
    progress: bool = False,
) -> dict[str, object]:
    """Compute the stock level joint regression described in the appendix."""
    repo_root = Path(repo_root).resolve()
    source_dir = Path(output_root).resolve()
    source_dir.mkdir(parents=True, exist_ok=True)
    stock = compute_stock_level_incremental_regression(
        repo_root,
        source_dir,
        start_ensemble=start_ensemble,
        max_ensemble=max_ensemble,
        jobs=max(1, int(jobs)),
        progress=progress,
    )
    return {"stock_level_rows": len(stock)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute appendix stock level joint regressions from saved forecasts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--output-root", default=str(SOURCE_DIR))
    parser.add_argument("--start-ensemble", type=int, default=0)
    parser.add_argument("--max-ensemble", type=int, default=9)
    parser.add_argument("--jobs", type=int, default=DEFAULT_SUPPLEMENTARY_JOBS)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    run_supplementary_experiments(
        repo_root=args.repo_root,
        output_root=args.output_root,
        start_ensemble=args.start_ensemble,
        max_ensemble=args.max_ensemble,
        jobs=args.jobs,
        progress=args.progress,
    )


if __name__ == "__main__":
    main()
