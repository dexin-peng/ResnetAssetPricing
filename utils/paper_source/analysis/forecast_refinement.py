"""Compute pairwise preservation/refinement and materialize renderer-ready caches.

Run separately from asset generation. No plotting or LaTeX writing occurs here.
Forecast inputs are already ensemble-averaged and aligned to a common base index.
"""
from __future__ import annotations

import os
for _option in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_option, "1")

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
from scipy.stats import rankdata

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils.paper_source.analysis.pairwise_ordering import prepare_anchor, pairwise_statistics_from_anchor
else:
    from .pairwise_ordering import prepare_anchor, pairwise_statistics_from_anchor

REPORT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = REPORT_DIR / "data/pairwise_refinement"
WIDTHS = (1, 2, 3, 4)
DEPTHS = tuple(range(6, 21))
DEPTH_GROUPS = (("Medium", 6, 10), ("Deep", 11, 20))
COMPARISON_SPECS = (("ResNet vs NN", "ResNet", "resnets", "NN", "nns"),
                    ("ResNet+ vs NN+", "ResNet+", "resnetps", "NN+", "nnps"))
MODELS = ("ResNet", "NN", "ResNet+", "NN+")
KEYS = ["Model", "Width Schedule", "Depth"]
METRICS = {"Preservation": ("P", 1.0), "Refinement": ("F", 100.0),
           "Value": ("V", 10000.0)}
DATES = tuple(pd.date_range("1987-01-31", "2023-11-30", freq="ME").strftime("%Y-%m-%d"))




def _boolean(values: pd.Series) -> pd.Series:
    parsed = values.astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})
    if parsed.isna().any():
        raise ValueError("Invalid boolean in monthly analysis input.")
    return parsed.astype(bool)


def prepare_monthly(monthly: pd.DataFrame) -> pd.DataFrame:
    """Normalize dates and derive the fixed all-month collapse sample."""
    frame = monthly.copy()
    frame["DATE"] = pd.to_datetime(frame["DATE"], errors="raise").dt.strftime("%Y-%m-%d")
    frame["Collapsed"] = _boolean(frame["Collapsed"])
    frame["Primary Eligible"] = ~frame.groupby(KEYS)["Collapsed"].transform("all")
    return frame.sort_values(KEYS + ["DATE"]).reset_index(drop=True)


def equal_cell_estimate(values: np.ndarray) -> tuple[float, np.ndarray, int]:
    """Average cell means; retain the calendar through a ratio influence for NA F."""
    values = np.asarray(values, dtype=float)
    months = len(values)
    counts = np.isfinite(values).sum(axis=0)
    active = counts > 0
    if not active.any():
        return float("nan"), np.full(months, np.nan), 0
    values, counts = values[:, active], counts[active]
    means = np.nansum(values, axis=0) / counts
    influence = np.where(np.isfinite(values), (values - means) / (counts / months), 0).mean(axis=1)
    return float(means.mean()), influence, int(active.sum())


def hac_mean_se(influence: np.ndarray, lags: int = 12) -> float:
    z = np.asarray(influence, dtype=float)
    if not np.isfinite(z).all():
        return float("nan")
    z = z - z.mean()
    n = len(z)
    variance = float(z @ z)
    for lag in range(1, min(lags, n - 1) + 1):
        variance += 2 * (1 - lag / (lags + 1)) * float(z[lag:] @ z[:-lag])
    return float(np.sqrt(max(variance, 0)) / n)


def _record(labels: dict, metric: str, value: float, influence: np.ndarray, specs: int) -> dict:
    scale = METRICS[metric][1]
    se = hac_mean_se(influence)
    return {**labels, "Metric": metric, "Estimate": value * scale,
        "StandardError": se * scale, "Lower": (value - 1.959963984540054 * se) * scale,
        "Upper": (value + 1.959963984540054 * se) * scale,
        "NW t-stat": value / se if se > 0 else float("nan"),
        "Specs": specs, "Months": len(influence), "Display Scale": scale}


def summarize_monthly(monthly: pd.DataFrame, *, include_collapsed: bool = False) -> dict[str, pd.DataFrame]:
    cells = {key: group.sort_values("DATE") for key, group in monthly.groupby(KEYS, sort=True)}
    selected = [key for key, part in cells.items() if include_collapsed or bool(part["Primary Eligible"].iloc[0])]
    summaries: dict[str, list[dict]] = {"group_summary": [], "depth_summary": [], "cell_summary": [], "matched_differences": []}

    def estimate(keys, metric):
        if not keys:
            return float("nan"), np.full(len(DATES), np.nan), 0
        return equal_cell_estimate(np.column_stack([cells[key][METRICS[metric][0]].to_numpy() for key in keys]))

    for key in cells:
        labels = dict(zip(KEYS, key)) | {"Primary Eligible": bool(cells[key]["Primary Eligible"].iloc[0])}
        for metric in METRICS:
            value, influence, specs = estimate([key], metric)
            summaries["cell_summary"].append(_record(labels, metric, value, influence, specs))

    scopes = [("group_summary", {"Depth Group": name}, set(range(lo, hi + 1))) for name, lo, hi in DEPTH_GROUPS]
    scopes += [("depth_summary", {"Depth": depth}, {depth}) for depth in DEPTHS]
    for scope, labels, depths in scopes:
        for comparison, left, _, right, _ in COMPARISON_SPECS:
            model_keys = {model: [k for k in selected if k[0] == model and k[2] in depths] for model in (left, right)}
            for metric in METRICS:
                model_values = {}
                for model, keys in model_keys.items():
                    value, influence, specs = estimate(keys, metric)
                    model_values[model] = (value, influence, specs)
                    row = _record({**labels, "Comparison": comparison, "Model": model}, metric, value, influence, specs)
                    if scope == "depth_summary":
                        row["Widths"] = len(keys)
                    summaries[scope].append(row)
                if scope == "group_summary":
                    a, b = model_values[left], model_values[right]
                    row = _record({**labels, "Comparison": comparison, "Model": "Difference"},
                        metric, a[0] - b[0], a[1] - b[1], min(a[2], b[2]))
                    row.update({"Residual Specs": a[2], "Benchmark Specs": b[2],
                                "Contrast": "difference of reported model means"})
                    summaries[scope].append(row)
                common = sorted(set(k[1:] for k in model_keys[left]) & set(k[1:] for k in model_keys[right]))
                common = [key for key in common if all(cells[(model, *key)][METRICS[metric][0]].notna().any() for model in (left, right))]
                if common:
                    a = estimate([(left, *key) for key in common], metric)
                    b = estimate([(right, *key) for key in common], metric)
                    summaries["matched_differences"].append(_record(
                        {**labels, "Scope": scope, "Comparison": comparison, "Model": "Difference"},
                        metric, a[0] - b[0], a[1] - b[1], len(common)))
    return {name: pd.DataFrame(rows) for name, rows in summaries.items()}


def _prepare_inputs(base_dir: Path, forecast_dirs: list[Path]) -> tuple[list[dict], dict[str, str]]:
    blocks = pd.read_csv(base_dir / "date_blocks.csv")
    arrays = {}
    for _, _, stem_a, _, stem_b in COMPARISON_SPECS:
        for width in WIDTHS:
            for depth in (width, *DEPTHS):
                for stem in (stem_a, stem_b):
                    prefix = f"{stem}{width}d{depth}"
                    path = next((folder / f"{prefix}.npy" for folder in forecast_dirs
                                 if (folder / f"{prefix}.npy").is_file()), None)
                    if path is None:
                        raise FileNotFoundError(f"Missing forecast array: {prefix}.npy")
                    arrays[prefix] = str(path.resolve())
    return blocks.to_dict("records"), arrays


def _compute_width(comparison_spec: tuple, width: int, base_path: str, blocks: list[dict], paths: dict[str, str]) -> pd.DataFrame:
    comparison, left, stem_a, right, stem_b = comparison_spec
    base = np.load(base_path, mmap_mode="r", allow_pickle=False)
    anchors = {model: np.load(paths[f"{stem}{width}d{width}"], mmap_mode="r", allow_pickle=False)
               for model, stem in ((left, stem_a), (right, stem_b))}
    prepared = []
    for row in blocks:
        start, end = int(row["Start"]), int(row["End"])
        valid = np.isfinite(base[start:end]).all(axis=1)
        realized = np.asarray(base[start:end, 0])[valid]
        prepared.append((row["DATE"], start, end, valid,
                         {m: prepare_anchor(np.asarray(a[start:end])[valid], realized) for m, a in anchors.items()}))
    rows = []
    for depth in DEPTHS:
        targets = {model: np.load(paths[f"{stem}{width}d{depth}"], mmap_mode="r", allow_pickle=False)
                   for model, stem in ((left, stem_a), (right, stem_b))}
        for date, start, end, valid, cached in prepared:
            for model, array in targets.items():
                target = np.asarray(array[start:end])[valid]
                stats = pairwise_statistics_from_anchor(cached[model], target)
                # Exact tie-aware all-pair payoff; unchanged pairs contribute zero.
                anchor = cached[model]
                stats["V"] = float(np.dot(
                    rankdata(target, method="average")
                    - rankdata(anchor.shallow.values, method="average"),
                    anchor.returns.values,
                ) / stats["total_pairs"])
                n_stocks = stats.pop("n_stocks")
                rows.append({"Comparison": comparison, "Model": model, "Width Schedule": width,
                    "Depth": depth, "Anchor Depth": width, "DATE": date, "N Stocks": n_stocks,
                    "Collapsed": bool(target.min() == target.max()), **stats})
    return pd.DataFrame(rows)


def compute_monthly(base_dir: Path, forecast_dirs: list[Path], jobs: int = 2) -> pd.DataFrame:
    blocks, paths = _prepare_inputs(base_dir, forecast_dirs)
    frames = []
    with ProcessPoolExecutor(max_workers=max(1, min(jobs, 8))) as pool:
        futures = {
            pool.submit(
                _compute_width,
                comparison_spec,
                width,
                str(base_dir / "base_panel.npy"),
                blocks,
                paths,
            ): (comparison_spec[0], width)
            for comparison_spec in COMPARISON_SPECS
            for width in WIDTHS
        }
        for future in as_completed(futures):
            frames.append(future.result())
            print(f"[pairwise-analysis] completed {futures[future]}", flush=True)
    return prepare_monthly(pd.concat(frames, ignore_index=True))


def write_cache(monthly: pd.DataFrame, cache_dir: Path) -> dict[str, pd.DataFrame]:
    monthly = prepare_monthly(monthly)
    summaries = summarize_monthly(monthly)
    all_cells = summarize_monthly(monthly, include_collapsed=True)["group_summary"]
    outputs = {"monthly": monthly, **summaries, "all_cells_summary": all_cells}
    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pairwise-", dir=cache_dir) as temp:
        staging = Path(temp)
        for name, frame in outputs.items():
            frame.to_csv(staging / f"{name}.csv", index=False)
        for path in staging.iterdir():
            os.replace(path, cache_dir / path.name)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, help="Common base and date-block export")
    parser.add_argument("--forecast-dir", type=Path, action="append", default=[], help="Search directory for averaged arrays; repeat for multiple directories")
    parser.add_argument("--monthly-input", type=Path, help="Rebuild summaries from an already computed monthly pairwise CSV")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    if args.monthly_input:
        if args.base_dir or args.forecast_dir:
            parser.error("Use either --monthly-input or the base/forecast inputs, not both.")
        monthly = prepare_monthly(pd.read_csv(args.monthly_input))
    else:
        if args.base_dir is None or not args.forecast_dir:
            parser.error("Full computation requires --base-dir and at least one --forecast-dir.")
        monthly = compute_monthly(args.base_dir.resolve(), [p.resolve() for p in args.forecast_dir], args.jobs)
    write_cache(monthly, args.cache_dir.resolve())
    primary_cells = monthly.loc[monthly["Primary Eligible"], KEYS].drop_duplicates().shape[0]
    print(f"[pairwise-analysis] wrote {len(monthly)} months across cells and {primary_cells} primary cells to {args.cache_dir}")


if __name__ == "__main__":
    main()
