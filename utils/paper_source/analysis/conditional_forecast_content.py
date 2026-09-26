from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPORT_DIR = Path(os.environ.get("RESASSETPRICING_REPORT_DIR", Path(__file__).resolve().parents[1]))
if str(REPORT_DIR) not in sys.path:
    sys.path.insert(0, str(REPORT_DIR))

from utils.paper_source.paths import REPO_ROOT, SOURCE_DIR
from utils.paper_source.analysis import supplementary_experiments as supplementary


RESIDUAL_MODEL = "ResNet+"
BENCHMARK_MODEL = "NN+"
EXPECTED_INPUT_CELLS = 74
EXPECTED_ESTIMABLE_CELLS = 70
EXPECTED_MONTHS = 443
EXPECTED_FIRST_MONTH = "1987-01-31"
EXPECTED_LAST_MONTH = "2023-11-30"

_WORKER_STATE: dict[str, object] = {}
_LOCAL_EXPORT_STATE: dict[str, object] = {}






def _init_worker(repo_root: str, start_ensemble: int, max_ensemble: int) -> None:
    root = Path(repo_root).resolve()
    X, y = supplementary._load_stock_level_inputs(root)
    _WORKER_STATE.clear()
    _WORKER_STATE.update(
        {
            "repo_root": root,
            "start_ensemble": int(start_ensemble),
            "max_ensemble": int(max_ensemble),
            "base_panel": supplementary._stock_level_base_panel(X, y),
        }
    )


def _monthly_coefficients_for_spec(spec: dict[str, object]) -> dict[str, object]:
    repo_root = _WORKER_STATE["repo_root"]
    start_ensemble = int(_WORKER_STATE["start_ensemble"])
    max_ensemble = int(_WORKER_STATE["max_ensemble"])
    base_panel = _WORKER_STATE["base_panel"]
    if not isinstance(repo_root, Path) or not isinstance(base_panel, pd.DataFrame):
        raise TypeError("Conditional forecast worker state was not initialized correctly.")

    residual = supplementary._load_forecast(
        repo_root,
        str(spec["residual_prefix"]),
        start_ensemble,
        max_ensemble,
    )
    benchmark = supplementary._load_forecast(
        repo_root,
        str(spec["control_prefix"]),
        start_ensemble,
        max_ensemble,
    )
    if residual.equals(benchmark):
        return {**spec, "monthly": pd.DataFrame(), "exclusion": "identical forecasts"}
    panel = pd.concat(
        [
            base_panel,
            residual.rename(RESIDUAL_MODEL),
            benchmark.rename(BENCHMARK_MODEL),
        ],
        axis=1,
        join="inner",
    ).dropna()
    reg_cols = [RESIDUAL_MODEL, BENCHMARK_MODEL] + list(supplementary.CONTROL_COLUMNS)
    rows: list[dict[str, object]] = []
    for date, block in panel.groupby(level="DATE", sort=True):
        if len(block) < len(reg_cols) + 20:
            continue
        standardized = supplementary._zscore(block, reg_cols).dropna()
        if len(standardized) < len(reg_cols) + 20:
            continue
        design = np.column_stack(
            [
                np.ones(len(standardized)),
                standardized[reg_cols].to_numpy(dtype=float),
            ]
        )
        if np.linalg.matrix_rank(design) != design.shape[1]:
            continue
        condition_number = float(np.linalg.cond(design))
        if not np.isfinite(condition_number) or condition_number >= 1.0e6:
            continue
        beta = np.linalg.lstsq(
            design,
            standardized["Realized return"].to_numpy(dtype=float),
            rcond=None,
        )[0]
        rows.append(
            {
                "DATE": pd.Timestamp(date),
                "ResNet+ Coefficient": float(beta[1]),
                "NN+ Coefficient": float(beta[2]),
                "N Stocks": int(len(standardized)),
                "Design Condition Number": condition_number,
            }
        )
    if not rows:
        raise ValueError(f"No monthly coefficients were estimated for s{spec['seed']}d{spec['depth']}.")
    return {**spec, "monthly": pd.DataFrame(rows)}


def _init_local_export_worker(export_dir: str) -> None:
    root = Path(export_dir).resolve()
    date_blocks = pd.read_csv(root / "date_blocks.csv")
    date_blocks["DATE"] = pd.to_datetime(date_blocks["DATE"], errors="raise")
    _LOCAL_EXPORT_STATE.clear()
    _LOCAL_EXPORT_STATE.update(
        {
            "export_dir": root,
            "base_panel": np.load(root / "base_panel.npy", mmap_mode="r", allow_pickle=False),
            "date_blocks": date_blocks,
        }
    )


def _local_export_worker(spec: dict[str, object]) -> dict[str, object]:
    export_dir = _LOCAL_EXPORT_STATE["export_dir"]
    base_panel = _LOCAL_EXPORT_STATE["base_panel"]
    date_blocks = _LOCAL_EXPORT_STATE["date_blocks"]
    if not isinstance(export_dir, Path) or not isinstance(base_panel, np.ndarray) or not isinstance(date_blocks, pd.DataFrame):
        raise TypeError("Local conditional forecast export state was not initialized correctly.")
    residual = np.load(
        export_dir / "forecasts" / f"{spec['residual_prefix']}.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    benchmark = np.load(
        export_dir / "forecasts" / f"{spec['control_prefix']}.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    if residual.shape != benchmark.shape or residual.shape[0] != base_panel.shape[0]:
        raise ValueError(f"Exported arrays have inconsistent shapes for s{spec['seed']}d{spec['depth']}.")
    if np.array_equal(residual, benchmark):
        return {**spec, "monthly": pd.DataFrame(), "exclusion": "identical forecasts"}

    rows: list[dict[str, object]] = []
    for date_row in date_blocks.itertuples(index=False):
        start = int(date_row.Start)
        end = int(date_row.End)
        base = np.asarray(base_panel[start:end], dtype=float)
        regressors = np.column_stack(
            [
                np.asarray(residual[start:end], dtype=float),
                np.asarray(benchmark[start:end], dtype=float),
                base[:, 1:],
            ]
        )
        realized = base[:, 0]
        valid = np.isfinite(realized) & np.isfinite(regressors).all(axis=1)
        regressors = regressors[valid]
        realized = realized[valid]
        if len(realized) < regressors.shape[1] + 20:
            continue
        means = regressors.mean(axis=0)
        standard_deviations = regressors.std(axis=0, ddof=0)
        if not np.isfinite(standard_deviations).all() or (standard_deviations <= 0.0).any():
            continue
        standardized = (regressors - means) / standard_deviations
        design = np.column_stack([np.ones(len(realized)), standardized])
        beta, _, rank, singular_values = np.linalg.lstsq(design, realized, rcond=None)
        if int(rank) != design.shape[1] or singular_values[-1] <= 0.0:
            continue
        condition_number = float(singular_values[0] / singular_values[-1])
        if not np.isfinite(condition_number) or condition_number >= 1.0e6:
            continue
        rows.append(
            {
                "DATE": pd.Timestamp(date_row.DATE),
                "ResNet+ Coefficient": float(beta[1]),
                "NN+ Coefficient": float(beta[2]),
                "N Stocks": int(len(realized)),
                "Design Condition Number": condition_number,
            }
        )
    if len(rows) != EXPECTED_MONTHS:
        raise ValueError(
            f"Expected {EXPECTED_MONTHS} monthly coefficients for s{spec['seed']}d{spec['depth']}, found {len(rows)}."
        )
    return {**spec, "monthly": pd.DataFrame(rows)}


def _records_to_frame(records: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for record in records:
        monthly = record["monthly"]
        if not isinstance(monthly, pd.DataFrame):
            raise TypeError("Each conditional forecast record must contain a monthly DataFrame.")
        if monthly.empty and record.get("exclusion") == "identical forecasts":
            continue
        for _, values in monthly.iterrows():
            rows.append(
                {
                    "Comparison": "ResNet+ vs NN+",
                    "Width Schedule": int(record["seed"]),
                    "Depth": int(record["depth"]),
                    "DATE": pd.Timestamp(values["DATE"]).strftime("%Y-%m-%d"),
                    "ResNet+ Coefficient": float(values["ResNet+ Coefficient"]),
                    "NN+ Coefficient": float(values["NN+ Coefficient"]),
                    "N Stocks": int(values["N Stocks"]),
                    "Design Condition Number": float(values["Design Condition Number"]),
                }
            )
    return pd.DataFrame(rows).sort_values(["Width Schedule", "Depth", "DATE"]).reset_index(drop=True)


def _validate_conditional_frame(out: pd.DataFrame) -> None:
    required = [
        "Width Schedule",
        "Depth",
        "DATE",
        "ResNet+ Coefficient",
        "NN+ Coefficient",
        "N Stocks",
        "Design Condition Number",
    ]
    if out.empty or out[required].isna().any().any():
        raise ValueError("Conditional forecast content output contains missing required values.")
    if out.duplicated(["Width Schedule", "Depth", "DATE"]).any():
        raise ValueError("Conditional forecast content output contains duplicate cell month rows.")
    if out[["Width Schedule", "Depth"]].drop_duplicates().shape[0] != EXPECTED_ESTIMABLE_CELLS:
        raise ValueError("Conditional forecast content output does not cover all estimable paired cells.")
    numeric_columns = [
        "ResNet+ Coefficient",
        "NN+ Coefficient",
        "N Stocks",
        "Design Condition Number",
    ]
    if not np.isfinite(out[numeric_columns].to_numpy()).all():
        raise ValueError("Conditional forecast content output contains nonfinite coefficients.")
    date_sequences = out.groupby(["Width Schedule", "Depth"], sort=True)["DATE"].agg(tuple)
    reference_dates = date_sequences.iloc[0]
    if (
        len(reference_dates) != EXPECTED_MONTHS
        or reference_dates[0] != EXPECTED_FIRST_MONTH
        or reference_dates[-1] != EXPECTED_LAST_MONTH
        or not date_sequences.map(lambda dates: dates == reference_dates).all()
    ):
        raise ValueError(
            "Every paired cell must cover the same 443 months from January 1987 "
            "through November 2023."
        )


def compute_conditional_forecast_content(
    repo_root: Path,
    source_dir: Path,
    output_path: Path,
    *,
    jobs: int,
    start_ensemble: int,
    max_ensemble: int,
) -> pd.DataFrame:
    repo_root = repo_root.resolve()
    specs = [
        spec
        for spec in supplementary._paired_incremental_specs(source_dir)
        if spec["residual_model"] == RESIDUAL_MODEL and spec["control_model"] == BENCHMARK_MODEL
    ]
    if len(specs) != EXPECTED_INPUT_CELLS:
        raise ValueError(f"Expected {EXPECTED_INPUT_CELLS} paired ResNet+ and NN+ cells, found {len(specs)}.")

    worker_count = max(1, min(int(jobs), len(specs)))
    records: list[dict[str, object]] = []
    with ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=_init_worker,
        initargs=(str(repo_root), int(start_ensemble), int(max_ensemble)),
    ) as executor:
        futures = {executor.submit(_monthly_coefficients_for_spec, spec): spec for spec in specs}
        for completed, future in enumerate(as_completed(futures), start=1):
            spec = futures[future]
            record = future.result()
            records.append(record)
            print(
                f"[conditional-content] {completed}/{len(specs)} "
                f"s{spec['seed']}d{spec['depth']}",
                flush=True,
            )

    out = _records_to_frame(records)
    _validate_conditional_frame(out)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return out


def compute_conditional_forecast_content_from_export(
    export_dir: Path,
    output_path: Path,
    *,
    jobs: int,
) -> pd.DataFrame:
    export_dir = export_dir.resolve()
    pairs = pd.read_csv(export_dir / "paired_cells.csv")
    required_pairs = {"Width Schedule", "Depth", "ResNet+ Prefix", "NN+ Prefix"}
    if not required_pairs.issubset(pairs.columns) or len(pairs) != EXPECTED_INPUT_CELLS:
        raise ValueError("Exported paired specification list is incomplete.")
    specs = [
        {
            "seed": int(row["Width Schedule"]),
            "depth": int(row["Depth"]),
            "residual_prefix": str(row["ResNet+ Prefix"]),
            "control_prefix": str(row["NN+ Prefix"]),
        }
        for _, row in pairs.iterrows()
    ]
    worker_count = max(1, min(int(jobs), len(specs)))
    records: list[dict[str, object]] = []
    with ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=_init_local_export_worker,
        initargs=(str(export_dir),),
    ) as executor:
        futures = {executor.submit(_local_export_worker, spec): spec for spec in specs}
        for completed, future in enumerate(as_completed(futures), start=1):
            spec = futures[future]
            records.append(future.result())
            print(
                f"[conditional-content-local] {completed}/{len(specs)} "
                f"s{spec['seed']}d{spec['depth']}",
                flush=True,
            )

    out = _records_to_frame(records)
    _validate_conditional_frame(out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    exclusions = sorted(
        [
        {
            "width_schedule": int(record["seed"]),
            "depth": int(record["depth"]),
            "reason": str(record["exclusion"]),
        }
        for record in records
        if record.get("exclusion")
        ],
        key=lambda item: (item["width_schedule"], item["depth"]),
    )
    expected_exclusions = [
        {"width_schedule": width, "depth": width, "reason": "identical forecasts"}
        for width in (1, 2, 3, 4)
    ]
    if exclusions != expected_exclusions:
        raise ValueError(f"Unexpected conditional forecast exclusions: {exclusions}.")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract monthly ResNet+ and NN+ coefficients from their joint stock return regression."
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--export-dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=SOURCE_DIR / "stock_level_conditional_forecast_content_monthly.csv",
    )
    parser.add_argument("--start-ensemble", type=int, default=0)
    parser.add_argument("--max-ensemble", type=int, default=9)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    if args.export_dir is None:
        out = compute_conditional_forecast_content(
            args.repo_root,
            args.source_dir,
            args.output,
            jobs=args.jobs,
            start_ensemble=args.start_ensemble,
            max_ensemble=args.max_ensemble,
        )
    else:
        out = compute_conditional_forecast_content_from_export(
            args.export_dir,
            args.output,
            jobs=args.jobs,
        )
    print(f"[conditional-content] wrote {len(out)} rows to {args.output}", flush=True)


if __name__ == "__main__":
    main()
