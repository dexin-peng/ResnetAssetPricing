from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path
import re
import time
import traceback

import numpy as np
import pandas as pd

FAMILY_BY_PREFIX = {
    "nns": "NN",
    "resnets": "ResNet",
    "nnps": "NN+",
    "resnetps": "ResNet+",
}
ASSET_PREFIX_RE = re.compile(r"^(resnetps|resnets|nnps|nns)(\d+)d(\d+)$")
DEFAULT_OUTPUT = Path("utils/paper_source/data/forecast_level_r2_variants.csv")

_REALIZED_RETURNS: pd.Series | None = None
_ASSET_DIR: Path | None = None


def _initializer(repo_root: str) -> None:
    global _REALIZED_RETURNS, _ASSET_DIR
    root = Path(repo_root)
    realized = pd.read_pickle(root / "tmp" / "clean_y.pkl")
    if not isinstance(realized, pd.Series):
        realized = realized.squeeze()
    _REALIZED_RETURNS = realized.rename("real")
    _ASSET_DIR = root / "asset"


def _tasks(asset_dir: Path) -> list[dict[str, object]]:
    tasks: list[dict[str, object]] = []
    for path in sorted(asset_dir.iterdir()):
        if not path.is_dir():
            continue
        match = ASSET_PREFIX_RE.match(path.name)
        if not match or not (path / "predicts").is_dir():
            continue
        prefix, seed, depth = match.groups()
        tasks.append(
            {
                "asset_prefix": path.name,
                "family": FAMILY_BY_PREFIX[prefix],
                "seed": int(seed),
                "depth": int(depth),
            }
        )
    return tasks


def _load_prediction(asset_prefix: str) -> pd.Series:
    if _ASSET_DIR is None:
        raise RuntimeError("worker was not initialized")
    pieces: list[pd.Series] = []
    base = _ASSET_DIR / asset_prefix / "predicts"
    for date_dir in sorted(path for path in base.iterdir() if path.is_dir()):
        total: pd.Series | None = None
        count = 0
        for file in sorted(date_dir.glob("predicts_*.pkl")):
            series = pd.read_pickle(file)
            if not isinstance(series, pd.Series):
                series = series.squeeze()
            series = series.astype("float64", copy=False)
            total = series.copy() if total is None else total.add(series, fill_value=0.0)
            count += 1
        if total is not None and count:
            pieces.append(total / float(count))
    if not pieces:
        raise FileNotFoundError(f"no prediction files for {asset_prefix}")
    out = pd.concat(pieces).sort_index()
    out.name = "pred"
    return out


def _r2_percent(predicted: np.ndarray, realized: np.ndarray) -> float:
    valid = np.isfinite(predicted) & np.isfinite(realized)
    if not valid.any():
        return np.nan
    predicted = predicted[valid]
    realized = realized[valid]
    realized_ss = np.sum(realized * realized)
    if realized_ss <= 0.0:
        return np.nan
    errors = predicted - realized
    return float(100.0 * (1.0 - np.sum(errors * errors) / realized_ss))


def _scaled_r2_percent(predicted: np.ndarray, realized: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(predicted) & np.isfinite(realized)
    if not valid.any():
        return np.nan, np.nan
    predicted = predicted[valid]
    realized = realized[valid]
    predicted_ss = np.sum(predicted * predicted)
    if predicted_ss <= 0.0:
        return np.nan, np.nan
    scale = float(np.sum(predicted * realized) / predicted_ss)
    if scale < 0.0:
        scale = 0.0
    return _r2_percent(scale * predicted, realized), scale


def _demean_by_date(predicted: np.ndarray, dates: pd.Index) -> np.ndarray:
    series = pd.Series(predicted, index=dates)
    means = series.groupby(level=0).transform("mean").to_numpy(dtype="float64")
    return predicted - means


def _rank_score_by_date(predicted: np.ndarray, dates: pd.Index) -> np.ndarray:
    series = pd.Series(predicted, index=dates)
    ranks = series.groupby(level=0).rank(method="average", pct=True).to_numpy(dtype="float64")
    return ranks - 0.5


def _decile_score_by_date(predicted: np.ndarray, dates: pd.Index) -> np.ndarray:
    series = pd.Series(predicted, index=dates)
    ranks = series.groupby(level=0).rank(method="average", pct=True)
    deciles = np.ceil(ranks * 10.0).clip(1.0, 10.0).to_numpy(dtype="float64")
    return (deciles - 5.5) / 4.5


def _tail_score_by_date(predicted: np.ndarray, dates: pd.Index, q: float = 0.10) -> np.ndarray:
    series = pd.Series(predicted, index=dates)
    ranks = series.groupby(level=0).rank(method="average", pct=True).to_numpy(dtype="float64")
    score = np.zeros_like(ranks)
    score[ranks <= q] = -1.0
    score[ranks >= 1.0 - q] = 1.0
    return score


def _trim_mask_by_prediction(prediction: pd.Series, dates: pd.Index, q: float) -> np.ndarray:
    grouped = prediction.groupby(level="DATE")
    lo_by_date = grouped.quantile(q)
    hi_by_date = grouped.quantile(1.0 - q)
    low = np.asarray(dates.map(lo_by_date), dtype="float64")
    high = np.asarray(dates.map(hi_by_date), dtype="float64")
    pred = prediction.to_numpy(dtype="float64")
    return (pred >= low) & (pred <= high)


def _compute_one(task: dict[str, object]) -> dict[str, object]:
    if _REALIZED_RETURNS is None:
        raise RuntimeError("worker was not initialized")
    prediction = _load_prediction(str(task["asset_prefix"]))
    frame = pd.concat([prediction, _REALIZED_RETURNS], axis=1, join="inner").dropna()
    frame.index = frame.index.set_names(["permno", "DATE"])

    pred_series = frame["pred"]
    pred = pred_series.to_numpy(dtype="float64")
    realized = frame["real"].to_numpy(dtype="float64")
    dates = frame.index.get_level_values("DATE")

    pred_dm = _demean_by_date(pred, dates)
    pred_rank = _rank_score_by_date(pred, dates)
    pred_decile = _decile_score_by_date(pred, dates)
    pred_tail = _tail_score_by_date(pred, dates)
    keep_5_each_tail = _trim_mask_by_prediction(pred_series, dates, 0.05)
    keep_2p5_each_tail = _trim_mask_by_prediction(pred_series, dates, 0.025)

    pred_5 = pred[keep_5_each_tail]
    realized_5 = realized[keep_5_each_tail]
    dates_5 = dates[keep_5_each_tail]
    pred_2p5 = pred[keep_2p5_each_tail]
    realized_2p5 = realized[keep_2p5_each_tail]
    dates_2p5 = dates[keep_2p5_each_tail]
    pred_5_dm = _demean_by_date(pred_5, dates_5)
    pred_2p5_dm = _demean_by_date(pred_2p5, dates_2p5)

    pred_dm_scaled_r2, pred_dm_scale = _scaled_r2_percent(pred_dm, realized)
    pred_5_dm_scaled_r2, pred_5_dm_scale = _scaled_r2_percent(pred_5_dm, realized_5)
    pred_2p5_dm_scaled_r2, pred_2p5_dm_scale = _scaled_r2_percent(pred_2p5_dm, realized_2p5)
    rank_scaled_r2, rank_scale = _scaled_r2_percent(pred_rank, realized)
    decile_scaled_r2, decile_scale = _scaled_r2_percent(pred_decile, realized)
    tail_scaled_r2, tail_scale = _scaled_r2_percent(pred_tail, realized)

    row = dict(task)
    row.update(
        {
            "n_obs": int(len(frame)),
            "n_obs_trim_5_each_tail": int(keep_5_each_tail.sum()),
            "n_obs_trim_2p5_each_tail": int(keep_2p5_each_tail.sum()),
            "raw_gu2020_r2oos": _r2_percent(pred, realized),
            "cs_demeaned_r2oos": _r2_percent(pred_dm, realized),
            "cs_demeaned_scaled_r2oos": pred_dm_scaled_r2,
            "cs_demeaned_scale": pred_dm_scale,
            "trim_5_each_tail_raw_r2oos": _r2_percent(pred_5, realized_5),
            "trim_2p5_each_tail_raw_r2oos": _r2_percent(pred_2p5, realized_2p5),
            "cs_demeaned_trim_5_each_tail_r2oos": _r2_percent(pred_5_dm, realized_5),
            "cs_demeaned_trim_5_each_tail_scaled_r2oos": pred_5_dm_scaled_r2,
            "cs_demeaned_trim_5_each_tail_scale": pred_5_dm_scale,
            "cs_demeaned_trim_2p5_each_tail_r2oos": _r2_percent(pred_2p5_dm, realized_2p5),
            "cs_demeaned_trim_2p5_each_tail_scaled_r2oos": pred_2p5_dm_scaled_r2,
            "cs_demeaned_trim_2p5_each_tail_scale": pred_2p5_dm_scale,
            "rank_scaled_r2oos": rank_scaled_r2,
            "rank_scale": rank_scale,
            "decile_scaled_r2oos": decile_scaled_r2,
            "decile_scale": decile_scale,
            "tail_10_scaled_r2oos": tail_scaled_r2,
            "tail_10_scale": tail_scale,
        }
    )
    return row


def _wrapped(task: dict[str, object]) -> dict[str, object]:
    try:
        return _compute_one(task)
    except Exception as exc:
        row = dict(task)
        row.update({"error": repr(exc), "traceback": traceback.format_exc()})
        return row


def compute_forecast_level_r2(
    *,
    repo_root: Path,
    output: Path,
    workers: int,
) -> pd.DataFrame:
    asset_dir = repo_root / "asset"
    tasks = _tasks(asset_dir)
    started = time.time()
    rows: list[dict[str, object]] = []
    with mp.Pool(processes=max(1, workers), initializer=_initializer, initargs=(str(repo_root),)) as pool:
        for index, row in enumerate(pool.imap_unordered(_wrapped, tasks), start=1):
            rows.append(row)
            status = "ERROR" if row.get("error") else "ok"
            print(
                f"{index}/{len(tasks)} {status} {row.get('asset_prefix')} "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )
    frame = pd.DataFrame(rows).sort_values(["family", "seed", "depth", "asset_prefix"])
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute forecast level R2 variants from saved predictions.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output = args.output if args.output.is_absolute() else repo_root / args.output
    frame = compute_forecast_level_r2(
        repo_root=repo_root,
        output=output,
        workers=args.workers,
    )
    errors = frame[frame["error"].notna()] if "error" in frame else pd.DataFrame()
    print(f"wrote={output} rows={len(frame)} errors={len(errors)}", flush=True)
    if not errors.empty:
        print(errors[["asset_prefix", "error"]].to_string(index=False), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
