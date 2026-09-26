"""Data preparation for the S1D7 diagnostic exhibit."""
from __future__ import annotations
import pandas as pd
from utils.paper_source.paths import SOURCE_DIR
MODEL_ORDER = ["nns1d3", "nns1d7", "resnets1d3", "resnets1d7"]

def _source_path(name: str):
    path = SOURCE_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Required S1D7 diagnostic source is missing: {path}")
    return path


def _read_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    activation = pd.read_csv(_source_path("s1d7_activation_panel.csv"))
    prediction = pd.read_csv(_source_path("s1d7_prediction_dispersion.csv"))
    econ = pd.read_csv(_source_path("s1d7_econ_summary.csv"))
    return activation, prediction, econ


def analyze_s1d7_diagnostic_summary_table() -> dict:
    activation, prediction, econ = _read_sources()
    pred_1987 = prediction[(prediction["date"].astype(str) == "19871231") & (prediction["ensemble"] == 0)].copy()
    act_hidden_1987 = activation[
        (activation["date"].astype(str) == "19871231")
        & (activation["ensemble"] == 0)
        & (activation["layer_name"] != "Input")
        & (activation["layer_name"] != "Output")
    ].copy()
    act_summary = (
        act_hidden_1987.groupby("model", sort=False)
        .agg(min_hidden_col_std=("mean_col_std", "min"), max_zero_var_frac=("zero_var_frac", "max"))
        .reset_index()
    )
    econ_all = econ[econ["group"] == "ALL"].copy()

    merged = (
        pd.DataFrame({"model": MODEL_ORDER})
        .merge(econ_all, on="model", how="left")
        .merge(pred_1987[["model", "prediction_std", "prediction_unique_12dp"]], on="model", how="left")
        .merge(act_summary, on="model", how="left")
    )

    return {"table": merged}
