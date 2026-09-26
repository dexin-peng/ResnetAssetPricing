"""Collect this checkout's economic evaluation CSVs without reading checkpoints."""
from pathlib import Path
from typing import Iterable
import argparse
import csv
import shutil
import tempfile
import pandas as pd
from .analysis.data import PLANNED_MAX_DEPTH, PUBLIC_WIDTH_SCHEDULES, REPORT_FAMILIES, normalize_decile_label, parse_prefix
from .paths import REPO_ROOT
DATE_BASES=("auto","formation","realization")
RESULT_FILENAMES=("brief_results.csv","econ_vw_decile_return.csv","econ_ew_decile_return.csv","econ_vw_turnover.csv","econ_ew_turnover.csv","econ_vw_decile_SR_table.csv")
def _source_file_record(relpath): return None
def _clean_value(value: str) -> str | float | int:
    text = str(value).strip()
    if text.endswith("%"):
        text = text[:-1]
    try:
        number = float(text)
    except ValueError:
        return value
    if number.is_integer():
        return int(number)
    return number

def _realized_return_dates(
    values: pd.Series, *, date_basis: str, turnover: bool, source: str,
) -> pd.Series:
    """Normalize only DATE labels; auto accepts the paper's complete calendars."""
    if date_basis not in DATE_BASES:
        raise ValueError(f"Unsupported date basis: {date_basis}")
    dates = pd.to_datetime(values.astype(str), errors="raise")
    if dates.isna().any():
        raise ValueError(f"Missing DATE in {source}.")
    basis = date_basis
    if basis == "auto":
        formation = pd.period_range("1987-02" if turnover else "1987-01", "2023-11", freq="M").to_timestamp("M")
        realization = formation + pd.offsets.MonthEnd(1)
        observed = pd.DatetimeIndex(dates.unique()).sort_values()
        if observed.equals(formation):
            basis = "formation"
        elif observed.equals(realization):
            basis = "realization"
        else:
            raise ValueError(
                f"Ambiguous or partial DATE calendar in {source}. Auto requires the paper's full "
                f"{'442-month turnover' if turnover else '443-month return'} calendar; "
                "pass --date-basis formation or --date-basis realization explicitly."
            )
    return dates + pd.offsets.MonthEnd(1) if basis == "formation" else dates

def _parse_brief(path: Path, relpath: str) -> dict[str, object] | None:
    parts = Path(relpath).parts
    if len(parts) < 5:
        return None
    prefix = parts[1]
    group = parts[3]
    parsed = parse_prefix(prefix)
    if parsed is None:
        return None
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    record: dict[str, object] = {
        "source_path": relpath,
        "group": group,
        **parsed,
    }
    for row in rows:
        if len(row) < 2:
            continue
        key = row[0].strip()
        value = row[1].strip()
        if key == "Brief Results":
            continue
        record[key] = _clean_value(value)
    return record

def _parse_decile_return(path: Path, relpath: str, *, date_basis: str = "auto") -> pd.DataFrame | None:
    parts = Path(relpath).parts
    if len(parts) < 5:
        return None
    prefix = parts[1]
    group = parts[3]
    parsed = parse_prefix(prefix)
    if parsed is None:
        return None
    df = pd.read_csv(path)
    if "DATE" not in df.columns:
        return None
    df["DATE"] = _realized_return_dates(df["DATE"], date_basis=date_basis, turnover=False, source=relpath)
    value_cols = [c for c in df.columns if c != "DATE"]
    out = df.melt(id_vars=["DATE"], value_vars=value_cols, var_name="decile", value_name="return")
    out["decile"] = out["decile"].map(normalize_decile_label)
    out.insert(0, "source_path", relpath)
    out.insert(1, "group", group)
    weighting = "EW" if Path(relpath).name == "econ_ew_decile_return.csv" else "VW"
    out.insert(2, "weighting", weighting)
    for key, value in reversed(list(parsed.items())):
        out.insert(3, key, value)
    out["asset_prefix"] = prefix
    return out

def _parse_decile_statistics(path: Path, relpath: str) -> pd.DataFrame | None:
    parts = Path(relpath).parts
    if len(parts) < 5:
        return None
    prefix = parts[1]
    group = parts[3]
    parsed = parse_prefix(prefix)
    if parsed is None:
        return None
    df = pd.read_csv(path)
    required = (
        "PredRetDecile",
        "Monthly Raw Preds Mean/%",
        "Monthly Excess Mean/%",
        "Monthly Std*100",
        "Sharpe Ratio",
    )
    if not set(required).issubset(df.columns):
        return None
    out = df[list(required)].copy().rename(
        columns={
            "PredRetDecile": "rank",
            "Monthly Raw Preds Mean/%": "pred",
            "Monthly Excess Mean/%": "avg",
            "Monthly Std*100": "sd",
            "Sharpe Ratio": "sr",
        }
    )
    out["rank"] = out["rank"].map(normalize_decile_label)
    for column in ("pred", "avg", "sd", "sr"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out.insert(0, "source_path", relpath)
    out.insert(1, "group", group)
    for key, value in reversed(list(parsed.items())):
        out.insert(2, key, value)
    out["asset_prefix"] = prefix
    return out

def _decile_statistics_panel(frames: list[pd.DataFrame]) -> pd.DataFrame:
    panel = pd.concat(frames, ignore_index=True)
    panel = panel[
        (panel["group"] == "ALL")
        & panel["family"].isin(REPORT_FAMILIES)
        & panel["seed"].isin(PUBLIC_WIDTH_SCHEDULES)
        & (pd.to_numeric(panel["depth"], errors="coerce") <= PLANNED_MAX_DEPTH)
    ].copy()
    rank_order = {str(rank): rank for rank in range(1, 11)} | {"LS": 11}
    panel["_rank_order"] = panel["rank"].map(rank_order)
    panel = panel.sort_values(["family", "seed", "depth", "_rank_order"]).drop(columns="_rank_order")
    return panel.reset_index(drop=True)

def _parse_turnover(path: Path, relpath: str, *, date_basis: str = "auto") -> pd.DataFrame | None:
    parts = Path(relpath).parts
    if len(parts) < 5:
        return None
    prefix = parts[1]
    group = parts[3]
    parsed = parse_prefix(prefix)
    if parsed is None:
        return None
    df = pd.read_csv(path)
    if "DATE" not in df.columns:
        return None
    df["DATE"] = _realized_return_dates(df["DATE"], date_basis=date_basis, turnover=True, source=relpath)
    value_cols = [c for c in df.columns if c != "DATE"]
    if not value_cols:
        return None
    value_col = "Turnover" if "Turnover" in value_cols else value_cols[0]
    out = df[["DATE", value_col]].copy().rename(columns={value_col: "turnover"})
    out.insert(0, "source_path", relpath)
    out.insert(1, "group", group)
    weighting = "EW" if Path(relpath).name == "econ_ew_turnover.csv" else "VW"
    out.insert(2, "weighting", weighting)
    for key, value in reversed(list(parsed.items())):
        out.insert(3, key, value)
    out["asset_prefix"] = prefix
    return out

def _build_coverage(brief: pd.DataFrame) -> pd.DataFrame:
    if brief.empty:
        return pd.DataFrame()
    all_brief = brief[brief["group"] == "ALL"].copy()
    if all_brief.empty:
        return pd.DataFrame()
    rows = []
    for (family, seed), block in all_brief.groupby(["family", "seed"], dropna=False):
        depths = sorted(pd.to_numeric(block["depth"], errors="coerce").dropna().astype(int).unique())
        if not depths:
            continue
        expected = max(0, PLANNED_MAX_DEPTH - int(seed) + 1)
        rows.append(
            {
                "family": family,
                "seed": int(seed),
                "completed_depths": len(depths),
                "expected_depths": expected,
                "min_depth": min(depths),
                "max_depth": max(depths),
                "missing_depths": " ".join(
                    str(d) for d in range(int(seed), PLANNED_MAX_DEPTH + 1) if d not in depths
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["family", "seed"]).reset_index(drop=True)

def collect_from_raw(
    raw_root: Path, relpaths: Iterable[str], output_root: Path, *, date_basis: str = "auto",
) -> None:
    relpaths = sorted(set(relpaths))
    brief_records: list[dict[str, object]] = []
    decile_frames: list[pd.DataFrame] = []
    decile_statistics_frames: list[pd.DataFrame] = []
    turnover_frames: list[pd.DataFrame] = []
    precomputed_turnover: pd.DataFrame | None = None
    source_files: dict[str, Path] = {}
    source_priorities: dict[str, int] = {}
    for relpath in relpaths:
        path = raw_root / relpath
        if not path.is_file():
            raise FileNotFoundError(f"Collected source is missing: {relpath}")
        source_record = _source_file_record(relpath)
        if source_record is not None:
            priority, name = source_record
            if name not in source_priorities or priority < source_priorities[name]:
                source_files[name] = path
                source_priorities[name] = priority
        elif relpath.endswith("/brief_results.csv"):
            record = _parse_brief(path, relpath)
            if record:
                brief_records.append(record)
        elif relpath.endswith(("/econ_vw_decile_return.csv", "/econ_ew_decile_return.csv")):
            frame = _parse_decile_return(path, relpath, date_basis=date_basis)
            if frame is not None and not frame.empty:
                decile_frames.append(frame)
        elif relpath.endswith("/econ_vw_decile_SR_table.csv"):
            frame = _parse_decile_statistics(path, relpath)
            if frame is not None and not frame.empty:
                decile_statistics_frames.append(frame)
        elif relpath.endswith(("/econ_vw_turnover.csv", "/econ_ew_turnover.csv")):
            frame = _parse_turnover(path, relpath, date_basis=date_basis)
            if frame is not None and not frame.empty:
                turnover_frames.append(frame)

    if not brief_records or not decile_frames:
        raise ValueError("No completed econ brief and decile return files were collected; existing sources were not replaced.")
    if "decile_portfolio_statistics.csv" in source_files:
        _validate_decile_statistics_source(source_files["decile_portfolio_statistics.csv"])
    if not turnover_frames and "turnover_panel.csv" in source_files:
        path = source_files["turnover_panel.csv"]
        precomputed_turnover = pd.read_csv(path)
        if not precomputed_turnover.empty:
            precomputed_turnover["DATE"] = _realized_return_dates(
                precomputed_turnover["DATE"], date_basis=date_basis, turnover=True, source=str(path),
            )
    output_root.mkdir(parents=True, exist_ok=True)
    for name, src in source_files.items():
        if name == "turnover_panel.csv":
            continue
        dst = output_root / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)

    brief = pd.DataFrame(brief_records)
    if not brief.empty:
        brief = brief.sort_values(["family", "seed", "depth", "group"]).reset_index(drop=True)
        brief_all = brief[brief["group"] == "ALL"].copy()
    else:
        brief_all = pd.DataFrame()
    brief.to_csv(output_root / "brief_results_panel.csv", index=False)
    brief_all.to_csv(output_root / "brief_results_all.csv", index=False)

    coverage = _build_coverage(brief)
    coverage.to_csv(output_root / "coverage_summary.csv", index=False)

    if decile_frames:
        decile = pd.concat(decile_frames, ignore_index=True)
        decile = decile.sort_values(["family", "seed", "depth", "group", "DATE", "decile"]).reset_index(drop=True)
    else:
        decile = pd.DataFrame()
    decile.to_csv(output_root / "decile_returns_panel.csv", index=False)

    if decile_statistics_frames:
        statistics = _decile_statistics_panel(decile_statistics_frames)
        if not statistics.empty:
            statistics.to_csv(output_root / "decile_portfolio_statistics.csv", index=False)

    if not turnover_frames and precomputed_turnover is not None and not precomputed_turnover.empty:
        turnover_frames.append(precomputed_turnover)

    if turnover_frames:
        turnover = pd.concat(turnover_frames, ignore_index=True)
        turnover = turnover.sort_values(["family", "seed", "depth", "group", "weighting", "DATE"]).reset_index(drop=True)
        turnover.to_csv(output_root / "turnover_panel.csv", index=False)
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',type=Path,default=REPO_ROOT)
    parser.add_argument('--output',type=Path,default=REPO_ROOT/'tmp/replication_inputs')
    parser.add_argument('--groups',nargs='+',default=['ALL','TOP80%','BOTTOM80%'])
    parser.add_argument('--date-basis',choices=DATE_BASES,default='auto')
    args=parser.parse_args()
    root=args.repo_root.resolve();output=args.output.resolve()
    paths=[]
    for family in ('nns','resnets','nnps','resnetps'):
        for width in PUBLIC_WIDTH_SCHEDULES:
            for depth in range(width,PLANNED_MAX_DEPTH+1):
                for group in args.groups:
                    for filename in RESULT_FILENAMES:
                        p=root/'asset'/f'{family}{width}d{depth}'/'econ'/group/filename
                        if p.is_file():paths.append(p.relative_to(root).as_posix())
    if not paths: raise FileNotFoundError('No completed econ outputs. Run training, prediction and economic evaluation first.')
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='resassetpricing-collect-',dir=output.parent) as directory:
        staging=Path(directory)
        collect_from_raw(root,paths,staging,date_basis=args.date_basis)
        # Write compressed public-format portfolio inputs without changing dates.
        decile=pd.read_csv(staging/'decile_returns_panel.csv')
        (staging/'decile_returns').mkdir()
        for i,start in enumerate(range(0,len(decile),500000)):
            decile.iloc[start:start+500000].to_csv(staging/'decile_returns'/f'part-{i:03d}.csv.gz',index=False)
        (staging/'decile_returns_panel.csv').unlink()
        turnover=staging/'turnover_panel.csv'
        if turnover.exists():
            pd.read_csv(turnover).to_csv(staging/'turnover_panel.csv.gz',index=False);turnover.unlink()
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f'{output} is not empty; choose a new --output directory for this run.')
        output.mkdir(parents=True,exist_ok=True)
        for path in staging.iterdir(): shutil.move(str(path),str(output/path.name))
    print(f'Collected {len(paths)} economic output files into {output}')
    print('Forecast diagnostics and theme analyses are separate; see utils/paper_source/README.md.')
if __name__=='__main__': main()
