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


PAIR_CONFIGS = {
    "resnet-nn": {
        "comparison": "ResNet vs NN",
        "residual_model": "ResNet",
        "benchmark_model": "NN",
        "residual_stem": "resnets",
        "benchmark_stem": "nns",
    },
    "resnetp-nnp": {
        "comparison": "ResNet+ vs NN+",
        "residual_model": "ResNet+",
        "benchmark_model": "NN+",
        "residual_stem": "resnetps",
        "benchmark_stem": "nnps",
    },
}
EXPECTED_WIDTH_SCHEDULES = (1, 2, 3, 4)
EXPECTED_CELLS = sum(21 - width for width in EXPECTED_WIDTH_SCHEDULES)
EXPECTED_PREFIXES = 148
EXPECTED_ROWS = 2_450_013
EXPECTED_MONTHS = 443
EXPECTED_FIRST_MONTH = "1987-01-31"
EXPECTED_LAST_MONTH = "2023-11-30"

_EXPORT_WORKER_STATE: dict[str, object] = {}






def _init_export_worker(
    repo_root: str,
    reference_index_path: str,
    sort_order_path: str,
    forecast_dir: str,
    start_ensemble: int,
    max_ensemble: int,
) -> None:
    _EXPORT_WORKER_STATE.clear()
    _EXPORT_WORKER_STATE.update(
        {
            "repo_root": Path(repo_root).resolve(),
            "reference_index": pd.read_pickle(reference_index_path),
            "sort_order": np.load(sort_order_path, mmap_mode="r", allow_pickle=False),
            "forecast_dir": Path(forecast_dir),
            "start_ensemble": int(start_ensemble),
            "max_ensemble": int(max_ensemble),
        }
    )


def _export_forecast(prefix: str) -> dict[str, object]:
    root = _EXPORT_WORKER_STATE["repo_root"]
    reference_index = _EXPORT_WORKER_STATE["reference_index"]
    sort_order = _EXPORT_WORKER_STATE["sort_order"]
    forecast_dir = _EXPORT_WORKER_STATE["forecast_dir"]
    if (
        not isinstance(root, Path)
        or not isinstance(reference_index, pd.Index)
        or not isinstance(sort_order, np.ndarray)
        or not isinstance(forecast_dir, Path)
    ):
        raise TypeError("Forecast export worker state was not initialized correctly.")
    forecast = supplementary._load_forecast(
        root,
        prefix,
        int(_EXPORT_WORKER_STATE["start_ensemble"]),
        int(_EXPORT_WORKER_STATE["max_ensemble"]),
    )
    if not forecast.index.equals(reference_index):
        raise ValueError(f"Forecast index for {prefix} does not match the reference stock month index.")
    values = forecast.to_numpy(dtype=np.float64)
    if len(values) != EXPECTED_ROWS or not np.isfinite(values).all():
        raise ValueError(f"Forecast values for {prefix} are incomplete or nonfinite.")
    output = forecast_dir / f"{prefix}.npy"
    np.save(output, values[sort_order], allow_pickle=False)
    return {
        "prefix": prefix,
        "relative_path": str(output.relative_to(forecast_dir.parent)),
    }


def export_conditional_forecast_inputs(
    repo_root: Path,
    source_dir: Path,
    export_dir: Path,
    *,
    pair_key: str,
    jobs: int,
    start_ensemble: int,
    max_ensemble: int,
) -> dict[str, object]:
    repo_root = repo_root.resolve()
    export_dir.mkdir(parents=True, exist_ok=True)
    forecast_dir = export_dir / "forecasts"
    forecast_dir.mkdir(parents=True, exist_ok=True)

    if pair_key not in PAIR_CONFIGS:
        raise ValueError(f"Unknown forecast pair {pair_key!r}.")
    pair_config = PAIR_CONFIGS[pair_key]
    residual_model = str(pair_config["residual_model"])
    benchmark_model = str(pair_config["benchmark_model"])
    brief_path = source_dir / "brief_results_all.csv"
    if not brief_path.exists():
        raise FileNotFoundError(f"Missing {brief_path}.")
    brief = pd.read_csv(brief_path)
    required_columns = {
        "family",
        "seed",
        "depth",
        "Asset Prefix",
        "Constant-Predict Decile Fallback Months",
    }
    missing_columns = sorted(required_columns.difference(brief.columns))
    if missing_columns:
        raise ValueError(
            "brief_results_all.csv is missing required forecast export columns: "
            + ", ".join(missing_columns)
        )
    for column in ("seed", "depth", "Constant-Predict Decile Fallback Months"):
        brief[column] = pd.to_numeric(brief[column], errors="coerce")
    brief = brief[
        brief["family"].isin((residual_model, benchmark_model))
        & brief["seed"].isin(EXPECTED_WIDTH_SCHEDULES)
    ].copy()
    planned_cells = {
        (width, depth)
        for width in EXPECTED_WIDTH_SCHEDULES
        for depth in range(width, 21)
    }
    model_rows: dict[str, dict[tuple[int, int], pd.Series]] = {}
    for model in (residual_model, benchmark_model):
        model_frame = brief[brief["family"].eq(model)].copy()
        if model_frame.duplicated(["seed", "depth"]).any():
            raise ValueError(f"brief_results_all.csv contains duplicate {model} width and depth rows.")
        rows = {
            (int(width), int(depth)): row
            for (width, depth), row in model_frame.set_index(["seed", "depth"]).iterrows()
            if pd.notna(width) and pd.notna(depth)
        }
        missing_cells = sorted(planned_cells.difference(rows))
        extra_cells = sorted(set(rows).difference(planned_cells))
        if missing_cells or extra_cells:
            raise ValueError(
                f"The {model} export grid is incomplete; missing={missing_cells}, extra={extra_cells}."
            )
        model_rows[model] = rows

    specs = []
    for width, depth in sorted(planned_cells):
        residual_row = model_rows[residual_model][(width, depth)]
        benchmark_row = model_rows[benchmark_model][(width, depth)]
        specs.append(
            {
                "seed": width,
                "depth": depth,
                "residual_prefix": str(residual_row["Asset Prefix"]),
                "benchmark_prefix": str(benchmark_row["Asset Prefix"]),
                "residual_collapse": bool(
                    float(residual_row["Constant-Predict Decile Fallback Months"]) > 0.0
                ),
                "benchmark_collapse": bool(
                    float(benchmark_row["Constant-Predict Decile Fallback Months"]) > 0.0
                ),
            }
        )
    if len(specs) != EXPECTED_CELLS:
        raise ValueError(f"Expected {EXPECTED_CELLS} paired forecast cells, found {len(specs)}.")
    pairs = pd.DataFrame(
        [
            {
                "Comparison": str(pair_config["comparison"]),
                "Residual Model": residual_model,
                "Benchmark Model": benchmark_model,
                "Width Schedule": int(spec["seed"]),
                "Depth": int(spec["depth"]),
                "Residual Prefix": str(spec["residual_prefix"]),
                "Benchmark Prefix": str(spec["benchmark_prefix"]),
                f"{residual_model} Prefix": str(spec["residual_prefix"]),
                f"{benchmark_model} Prefix": str(spec["benchmark_prefix"]),
                "Residual Constant Forecast": bool(spec["residual_collapse"]),
                "Benchmark Constant Forecast": bool(spec["benchmark_collapse"]),
            }
            for spec in specs
        ]
    ).sort_values(["Width Schedule", "Depth"])
    if pairs.duplicated(["Width Schedule", "Depth"]).any():
        raise ValueError("Paired specification list contains duplicate width and depth cells.")
    pairs_path = export_dir / "paired_cells.csv"
    pairs.to_csv(pairs_path, index=False)

    prefixes = sorted(set(pairs["Residual Prefix"]) | set(pairs["Benchmark Prefix"]))
    if len(prefixes) != EXPECTED_PREFIXES:
        raise ValueError(f"Expected {EXPECTED_PREFIXES} unique forecast prefixes, found {len(prefixes)}.")
    reference = supplementary._load_forecast(repo_root, prefixes[0], start_ensemble, max_ensemble)
    if len(reference) != EXPECTED_ROWS or reference.index.has_duplicates:
        raise ValueError("Reference forecast does not contain the expected unique stock month rows.")
    dates = pd.to_datetime(reference.index.get_level_values("DATE"))
    unique_dates = pd.Index(dates.unique()).sort_values()
    if (
        len(unique_dates) != EXPECTED_MONTHS
        or unique_dates[0] != pd.Timestamp(EXPECTED_FIRST_MONTH)
        or unique_dates[-1] != pd.Timestamp(EXPECTED_LAST_MONTH)
    ):
        raise ValueError("Reference forecast does not cover the expected 443 evaluation months.")
    sort_order = np.argsort(dates.to_numpy(), kind="stable")
    sort_order_path = export_dir / "_sort_order.npy"
    np.save(sort_order_path, sort_order, allow_pickle=False)
    worker_reference_index_path = export_dir / "_reference_index_unsorted.pkl"
    pd.to_pickle(reference.index, worker_reference_index_path)
    sorted_index = reference.index[sort_order]
    reference_index_path = export_dir / "reference_index.pkl"
    pd.to_pickle(sorted_index, reference_index_path)
    sorted_dates = pd.to_datetime(sorted_index.get_level_values("DATE"))
    date_counts = pd.Series(sorted_dates).value_counts(sort=True).sort_index()
    date_blocks = pd.DataFrame(
        {
            "DATE": date_counts.index,
            "N Rows": date_counts.to_numpy(dtype=int),
        }
    )
    date_blocks["End"] = date_blocks["N Rows"].cumsum()
    date_blocks["Start"] = date_blocks["End"] - date_blocks["N Rows"]
    date_blocks = date_blocks[["DATE", "Start", "End", "N Rows"]]
    date_blocks_path = export_dir / "date_blocks.csv"
    date_blocks.to_csv(date_blocks_path, index=False)

    X, y = supplementary._load_stock_level_inputs(repo_root)
    base = pd.concat(
        [y.rename("Realized return"), X[supplementary.CONTROL_COLUMNS]],
        axis=1,
        join="inner",
    ).reindex(reference.index)
    base_columns = ["Realized return"] + list(supplementary.CONTROL_COLUMNS)
    base_values = base[base_columns].to_numpy(dtype=np.float64)
    if base_values.shape != (EXPECTED_ROWS, len(base_columns)):
        raise ValueError("Base return and characteristic panel has an unexpected shape.")
    base_path = export_dir / "base_panel.npy"
    np.save(base_path, base_values[sort_order], allow_pickle=False)
    del X, y, base, base_values

    worker_count = max(1, min(int(jobs), len(prefixes)))
    forecast_records: list[dict[str, object]] = []
    with ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=_init_export_worker,
        initargs=(
            str(repo_root),
            str(worker_reference_index_path),
            str(sort_order_path),
            str(forecast_dir),
            int(start_ensemble),
            int(max_ensemble),
        ),
    ) as executor:
        futures = {executor.submit(_export_forecast, prefix): prefix for prefix in prefixes}
        for completed, future in enumerate(as_completed(futures), start=1):
            prefix = futures[future]
            record = future.result()
            forecast_records.append(record)
            print(f"[forecast-pair-export] {completed}/{len(prefixes)} {prefix}", flush=True)

    file_list = ["paired_cells.csv", "reference_index.pkl", "date_blocks.csv", "base_panel.npy"]
    file_list.extend(str(item["relative_path"]) for item in sorted(forecast_records, key=lambda item: str(item["prefix"])))
    (export_dir / "files-from.txt").write_text("\n".join(file_list) + "\n", encoding="utf-8")
    return {"forecast_prefixes": len(prefixes), "comparison": str(pair_config["comparison"])}



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a compact forecast pair bundle for local refinement analysis."
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument(
        "--pair-key",
        choices=tuple(PAIR_CONFIGS),
        default="resnetp-nnp",
    )
    parser.add_argument("--start-ensemble", type=int, default=0)
    parser.add_argument("--max-ensemble", type=int, default=9)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    result = export_conditional_forecast_inputs(
        args.repo_root,
        args.source_dir,
        args.export_dir,
        pair_key=args.pair_key,
        jobs=args.jobs,
        start_ensemble=args.start_ensemble,
        max_ensemble=args.max_ensemble,
    )
    print(
        f"[forecast-pair-export] wrote {result['forecast_prefixes']} forecast arrays "
        f"for {result['comparison']} to {args.export_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
