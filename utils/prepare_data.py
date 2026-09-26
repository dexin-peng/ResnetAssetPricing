#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.preprocess_handler import PreprocessHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build cleaned ResAssetPricing inputs under ./tmp from raw files under ./source_data."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root. Defaults to the checkout containing utils/.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild caches even if the target files already exist.",
    )
    return parser.parse_args()


def ensure_required_paths(project_root: Path) -> tuple[Path, Path, Path, Path, Path]:
    source_dir = project_root / "source_data"
    tmp_dir = project_root / "tmp"
    price_path = source_dir / "datashare_with_return.pkl"
    macro_path = source_dir / "PredictorData2024.xlsx"
    ff5_path = source_dir / "F-F_Research_Data_5_Factors_2x3.csv"
    description_path = source_dir / "Company94Characteristics.csv"

    required = (price_path, macro_path, ff5_path)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required raw input file(s): {', '.join(missing)}")

    if description_path.exists():
        print(f"[prepare_data] Found optional metadata file: {description_path}")

    tmp_dir.mkdir(parents=True, exist_ok=True)
    return price_path, macro_path, ff5_path, description_path, tmp_dir


def load_price_characteristics(price_path: Path) -> pd.DataFrame:
    print(f"[prepare_data] Loading firm characteristics from {price_path}")
    price_characteristics = pd.read_pickle(price_path)

    if "ret_exc_lead1m" in price_characteristics.columns and "RET" not in price_characteristics.columns:
        price_characteristics = price_characteristics.rename(columns={"ret_exc_lead1m": "RET"})

    if "RET" not in price_characteristics.columns:
        raise KeyError("Expected either 'ret_exc_lead1m' or 'RET' in datashare_with_return.pkl")

    if "me" in price_characteristics.columns and "market_equity" in price_characteristics.columns:
        same_market_cap = price_characteristics["me"].equals(price_characteristics["market_equity"])
        if not same_market_cap:
            me = price_characteristics["me"]
            market_equity = price_characteristics["market_equity"]
            same_market_cap = me.fillna(0).equals(market_equity.fillna(0))
        if same_market_cap:
            price_characteristics = price_characteristics.drop(columns=["me"])

    required_columns = {"permno", "DATE", "market_equity"}
    missing = required_columns.difference(price_characteristics.columns)
    if missing:
        raise KeyError(f"Missing required columns in datashare_with_return.pkl: {sorted(missing)}")

    if "me" in price_characteristics:
        raise ValueError("The duplicated me column differs from market_equity; resolve the source columns before preprocessing.")
    if price_characteristics[["permno", "DATE"]].isna().any().any():
        raise ValueError("Missing stock identifiers or dates.")
    if price_characteristics.duplicated(["permno", "DATE"]).any():
        raise ValueError("Duplicate stock-month observations.")
    if price_characteristics["RET"].isna().any():
        raise ValueError("Missing realized returns: remove unavailable targets before preparing inputs; targets must not be imputed.")
    price_characteristics["DATE"] = pd.to_datetime(price_characteristics["DATE"])
    price_characteristics = price_characteristics.set_index(["permno", "DATE"]).sort_index()
    price_characteristics = price_characteristics.loc[(slice(None), slice("1963-01-01", "2023-12-31")), :]
    print(f"[prepare_data] Firm panel shape after date filter: {price_characteristics.shape}")
    return price_characteristics


def load_macro_predictors(macro_path: Path) -> pd.DataFrame:
    print(f"[prepare_data] Loading macro predictors from {macro_path}")
    monthly = pd.read_excel(macro_path, sheet_name="Monthly")

    monthly.columns = [str(column).strip() for column in monthly.columns]
    lower_map = {column.lower(): column for column in monthly.columns}

    def _series_for(*candidates: str) -> pd.Series:
        for candidate in candidates:
            actual = lower_map.get(candidate.lower())
            if actual is not None:
                return pd.to_numeric(monthly[actual], errors="coerce")
        raise KeyError(f"Missing required monthly macro column. Tried: {candidates}")

    monthly["DATE"] = pd.to_datetime(_series_for("yyyymm").astype(int).astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)

    selected = pd.DataFrame(
        {
            "DATE": monthly["DATE"],
            "d/p": _series_for("d/p") if "d/p" in lower_map else np.log(_series_for("d12")) - np.log(_series_for("price", "index")),
            "e/p": _series_for("e/p") if "e/p" in lower_map else np.log(_series_for("e12")) - np.log(_series_for("price", "index")),
            "b/m": _series_for("b/m"),
            "ntis": _series_for("ntis"),
            "tbl": _series_for("tbl"),
            "tms": _series_for("tms") if "tms" in lower_map else _series_for("lty") - _series_for("tbl"),
            "dfy": _series_for("dfy") if "dfy" in lower_map else _series_for("baa") - _series_for("aaa"),
            "svar": _series_for("svar"),
        }
    ).set_index("DATE").sort_index()
    selected = selected.loc["1963-01-01":"2023-12-31"]
    print(f"[prepare_data] Monthly macro panel shape after filter: {selected.shape}")
    return selected


def load_ff5(ff5_path: Path) -> pd.DataFrame:
    print(f"[prepare_data] Loading FF5 factors from {ff5_path}")
    # Accept both the original French download and the header-only monthly CSV.
    import io
    lines = ff5_path.read_text(encoding="utf-8-sig").splitlines()
    header_index = next((i for i, line in enumerate(lines) if "Mkt-RF" in line and "SMB" in line and "RF" in line), None)
    if header_index is None:
        raise ValueError("Cannot find the FF5 monthly header (Mkt-RF, SMB, HML, RMW, CMA, RF).")
    columns = [part.strip() for part in lines[header_index].split(",")]
    monthly_rows = []
    for line in lines[header_index + 1:]:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == len(columns) and len(parts[0]) == 6 and parts[0].isdigit():
            monthly_rows.append(parts)
    ff5 = pd.DataFrame(monthly_rows, columns=["yyyymm"] + columns[1:])
    ff5["yyyymm"] = pd.to_numeric(ff5["yyyymm"], errors="raise")
    ff5 = ff5.loc[ff5["yyyymm"].between(196307, 202312)].copy()
    ff5["DATE"] = pd.to_datetime(ff5["yyyymm"].astype(int).astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
    ff5 = ff5.drop(columns=["yyyymm"]).set_index("DATE").sort_index().apply(pd.to_numeric, errors="raise")
    required = {"Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"}
    if not required.issubset(ff5.columns) or ff5.empty or ff5.index.duplicated().any():
        raise ValueError("Incomplete or duplicated FF5 monthly data.")
    print(f"[prepare_data] FF5 panel shape after filter: {ff5.shape}")
    return ff5


def build_clean_caches(project_root: Path, force: bool) -> None:
    price_path, macro_path, ff5_path, _, tmp_dir = ensure_required_paths(project_root)

    clean_x_path = tmp_dir / "clean_X.pkl"
    clean_y_path = tmp_dir / "clean_y.pkl"
    clean_mvel1_path = tmp_dir / "clean_mvel1.pkl"
    clean_ff5_path = tmp_dir / "clean_ff5.pkl"

    targets = (clean_x_path, clean_y_path, clean_mvel1_path, clean_ff5_path)
    if not force and all(path.exists() for path in targets):
        print("[prepare_data] Clean caches already exist. Use --force to rebuild them.")
        return

    price_characteristics = load_price_characteristics(price_path)
    macro = load_macro_predictors(macro_path)
    ff5 = load_ff5(ff5_path)

    preprocess_handler = PreprocessHandler(price_characteristics)
    preprocess_handler.fill_missing_value_with_median

    y = price_characteristics.pop("RET")
    mvel1 = price_characteristics["market_equity"].copy()

    price_characteristics.loc[:, :] = price_characteristics.groupby(level="DATE").transform(
        lambda s: s.rank(pct=True) * 2 - 1
    )
    price_characteristics = price_characteristics.fillna(0)

    if price_characteristics.shape[1] != 153:
        raise ValueError(f"Expected 153 firm characteristics, found {price_characteristics.shape[1]}.")
    if macro.index.duplicated().any():
        raise ValueError("Duplicate monthly macro observations.")
    x = price_characteristics.join(macro, how="left")
    if not np.isfinite(x.to_numpy()).all() or not np.isfinite(y.to_numpy()).all():
        raise ValueError("Nonfinite predictors or returns.")

    if x.isna().sum().sum() != 0:
        missing_columns = x.columns[x.isna().any()].tolist()
        raise ValueError(
            f"Joined feature matrix still contains missing values. Problematic columns: {missing_columns}"
        )
    if y.isna().sum() != 0:
        raise ValueError("Target series still contains missing values after preprocessing.")

    print(f"[prepare_data] Saving cleaned caches under {tmp_dir}")
    x.to_pickle(clean_x_path)
    y.to_pickle(clean_y_path)
    mvel1.to_pickle(clean_mvel1_path)
    ff5.to_pickle(clean_ff5_path)
    print("[prepare_data] Done.")


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    build_clean_caches(project_root, force=args.force)


if __name__ == "__main__":
    main()
