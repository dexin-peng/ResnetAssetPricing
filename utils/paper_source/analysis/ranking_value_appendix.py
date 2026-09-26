"""Compute appendix-only ranking-value levels and theme losses.

Run locally: uv run --with numpy --with pandas --with scipy
python -m utils.paper_source.analysis.ranking_value_appendix. Main-text sources are read-only.
Completed anchor losses may be reused; use --refresh-anchors after changing
forecast inputs. Missing anchor caches are computed from local zeroed forecasts.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

from utils.paper_source.analysis.forecast_refinement import equal_cell_estimate, hac_mean_se
from utils.paper_source.analysis.theme_refinement import MODELS, holm
from utils.paper_source.paths import REPO_ROOT, SOURCE_DIR

INPUTS = REPO_ROOT / "utils/paper_source/local_inputs/conditional_content_inputs"
PAIRED = REPO_ROOT / "utils/paper_source/local_inputs/theme_refinement"
PRIVATE = REPO_ROOT / "utils/paper_source/local_inputs/ranking_value_appendix"
LEVEL_SOURCE = SOURCE_DIR / "appendix_ranking_value_levels.csv"
LOSS_SOURCE = SOURCE_DIR / "appendix_theme_value_losses.csv"
GROUPS = (("Medium", 6, 10), ("Deep", 11, 20))
COMPARISONS = (("ResNet vs NN", "ResNet", "NN"), ("ResNet+ vs NN+", "ResNet+", "NN+"))


def ranking_value(scores, returns):
    size = len(returns)
    return float((rankdata(scores, method="average") - (size + 1) / 2) @ returns
                 / (size * (size - 1) / 2))


def original_anchor(model, width):
    directory = INPUTS / "forecasts" if model.endswith("+") else REPO_ROOT / "utils/paper_source/local_inputs/baseline"
    return np.load(directory / f"{MODELS[model]}{width}d{width}.npy", mmap_mode="r")


def estimate(frame, metric, dates):
    matrix = frame.pivot(index="DATE", columns=["Width Schedule", "Depth"], values=metric).reindex(dates)
    assert matrix.shape[0] == 443 and np.isfinite(matrix.to_numpy()).all()
    return equal_cell_estimate(matrix.to_numpy())


def record(labels, mean, influence, specs):
    se = hac_mean_se(influence)
    t = mean / se if se else (0.0 if mean == 0 else np.sign(mean) * np.inf)
    return dict(labels, Estimate=10000 * mean, StandardError=10000 * se,
                Lower=10000 * (mean - 1.95996398454 * se),
                Upper=10000 * (mean + 1.95996398454 * se),
                **{"NW t-stat": t, "Display Scale": 10000},
                PValue=2 * norm.sf(abs(t)), Specs=specs, Months=len(influence))


def full_input_levels(monthly, blocks, base):
    anchors = []
    for model in MODELS:
        for width in range(1, 5):
            forecast = original_anchor(model, width)
            assert len(forecast) == len(base) and np.isfinite(forecast).all()
            for row in blocks.itertuples(index=False):
                start, end = int(row.Start), int(row.End)
                assert np.isfinite(base[start:end]).all()
                anchors.append({"Model": model, "Width Schedule": width, "DATE": row.DATE,
                                "Shallow": ranking_value(forecast[start:end], base[start:end, 0]),
                                "N Stocks_anchor": end - start})
    data = monthly.merge(pd.DataFrame(anchors), on=["Model", "Width Schedule", "DATE"],
                         validate="many_to_one")
    assert len(data) == len(monthly) and data["N Stocks"].eq(data["N Stocks_anchor"]).all()
    data["Deep"] = data.Shallow + data.V
    records = []
    for group, lo, hi in GROUPS:
        subset = data[data.Depth.between(lo, hi)]
        for comparison, left, right in COMPARISONS:
            estimates = {}
            for model in (left, right):
                for metric in ("Shallow", "Deep"):
                    result = estimate(subset[subset.Model.eq(model)], metric, blocks.DATE)
                    estimates[model, metric] = result
                    records.append(record({"Depth Group": group, "Comparison": comparison,
                                           "Model": model, "Metric": metric}, *result))
            for metric in ("Shallow", "Deep"):
                a, b = estimates[left, metric], estimates[right, metric]
                row = record({"Depth Group": group, "Comparison": comparison,
                              "Model": "Difference", "Metric": metric},
                             a[0] - b[0], a[1] - b[1], min(a[2], b[2]))
                row.update({"Residual Specs": a[2], "Benchmark Specs": b[2],
                            "Contrast": "difference of reported model means"})
                records.append(row)
    result = pd.DataFrame(records)
    _check_levels(result)
    result.to_csv(LEVEL_SOURCE, index=False)
    print("PASS: 24 appendix level entries reconcile with all 12 Table 4 V entries.", flush=True)


def anchor_losses(model, width, themes, blocks, base, refresh):
    path = PRIVATE / "anchors" / f"{MODELS[model]}{width}d{width}.csv.gz"
    if refresh or not path.exists():
        reference = pd.read_pickle(INPUTS / "reference_index.pkl")
        original = original_anchor(model, width)
        records = []
        prefix = f"{MODELS[model]}{width}d{width}"
        for theme in themes:
            slug = theme.lower().replace(" ", "_").replace("-", "_")
            frame = pd.read_pickle(REPO_ROOT / f"asset/{prefix}/econ/ALL/group_ablation/{slug}/stock_predictions.pkl.gz")
            assert frame.attrs["model"] == prefix and frame.attrs["theme"] == theme
            assert (frame.attrs["start_ensemble"], frame.attrs["max_ensemble"]) == (0, 9)
            assert len(frame) == len(reference) and not frame.index.has_duplicates
            order = frame.index.get_indexer(reference)
            assert (order >= 0).all() and np.array_equal(frame.RealRet.to_numpy()[order], base[:, 0])
            masked = frame.PredRet.to_numpy()[order]
            assert np.isfinite(masked).all()
            for row in blocks.itertuples(index=False):
                start, end = int(row.Start), int(row.End)
                r = base[start:end, 0]
                loss = ranking_value(original[start:end], r) - ranking_value(masked[start:end], r)
                records.append(dict(Model=model, Width=width, DATE=row.DATE, Theme=theme, V=loss))
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(path, index=False)
    frame = pd.read_csv(path)
    assert len(frame) == 13 * 443 and not frame.duplicated(["DATE", "Theme"]).any(), path
    assert frame.Model.eq(model).all() and frame.Width.eq(width).all(), path
    assert set(frame.Theme) == set(themes) and set(frame.DATE) == set(blocks.DATE), path
    assert np.isfinite(frame.V).all(), path
    return frame[["DATE", "Theme", "V"]].rename(columns={"V": "Shallow"})


def theme_losses(monthly, blocks, base, refresh):
    main = pd.read_csv(SOURCE_DIR / "theme_pairwise_loss_summary.csv")
    assert main.AnchorCondition.eq("same_theme_zeroed").all()
    themes = sorted(main.Theme.unique())
    rows = []
    for model, stem in MODELS.items():
        specs = monthly[monthly.Model.eq(model) & monthly.Depth.between(11, 20)][
            ["Width Schedule", "Depth"]].drop_duplicates()
        cells = []
        for width, group in specs.groupby("Width Schedule"):
            shallow = anchor_losses(model, int(width), themes, blocks, base, refresh)
            for depth in group.Depth:
                path = PAIRED / "cells" / f"{stem}{int(width)}d{int(depth)}.csv.gz"
                frame = pd.read_csv(path, usecols=["Model", "Width", "Depth", "DATE", "Theme", "V"])
                assert len(frame) == 13 * 443 and not frame.duplicated(["DATE", "Theme"]).any(), path
                assert frame.Model.eq(model).all() and frame.Width.eq(width).all() and frame.Depth.eq(depth).all()
                frame = frame.merge(shallow, on=["DATE", "Theme"], validate="one_to_one")
                assert len(frame) == 13 * 443
                frame["Deep"] = frame.Shallow + frame.V
                cells.append(frame.rename(columns={"Width": "Width Schedule"}))
        data = pd.concat(cells, ignore_index=True)
        for theme, group in data.groupby("Theme"):
            for metric in ("Shallow", "Deep"):
                result = estimate(group, metric, blocks.DATE)
                assert result[2] == len(specs)
                rows.append(record({"Model": model, "Theme": theme, "Metric": metric}, *result))
        print("checked appendix theme losses", model, len(specs), "specifications", flush=True)
    result = pd.DataFrame(rows)
    result["Holm PValue"] = result.groupby(["Model", "Metric"]).PValue.transform(holm)
    result["AnchorCondition"] = "same_theme_zeroed"
    _check_losses(result, main)
    result.to_csv(LOSS_SOURCE, index=False)
    print("PASS: 104 appendix losses reconcile with all 52 Table 5 V losses.", flush=True)


def _check_levels(frame):
    keys = ["Depth Group", "Comparison", "Model", "Metric"]
    assert len(frame) == 24 and not frame.duplicated(keys).any()
    assert set(frame.Metric) == {"Shallow", "Deep"} and frame.Months.eq(443).all()
    assert np.isfinite(frame[["Estimate", "StandardError", "PValue"]]).all().all()
    wide = frame.pivot(index=keys[:3], columns="Metric", values="Estimate")
    main = pd.read_csv(SOURCE_DIR / "pairwise_refinement/group_summary.csv")
    target = main[main.Metric.eq("Value")].set_index(keys[:3])
    assert np.allclose(wide.Deep - wide.Shallow, target.loc[wide.index, "Estimate"], rtol=0, atol=1e-10)


def _check_losses(frame, main):
    keys = ["Model", "Theme", "Metric"]
    assert len(frame) == 104 and not frame.duplicated(keys).any()
    assert set(frame.Model) == set(MODELS) and set(frame.Metric) == {"Shallow", "Deep"}
    assert set(frame.Theme) == set(main.Theme) and frame.groupby(["Model", "Metric"]).size().eq(13).all()
    assert frame.Months.eq(443).all() and frame.AnchorCondition.eq("same_theme_zeroed").all()
    assert np.isfinite(frame[["Estimate", "StandardError", "Holm PValue"]]).all().all()
    assert frame["Holm PValue"].between(0, 1).all()
    wide = frame.pivot(index=["Model", "Theme"], columns="Metric", values="Estimate")
    target = main[main.Metric.eq("V")].set_index(["Model", "Theme"])
    assert np.allclose(wide.Deep - wide.Shallow, target.loc[wide.index, "display_mean"], rtol=0, atol=1e-10)
    for _, row in frame.iterrows():
        assert row.Specs == target.loc[(row.Model, row.Theme), "Specs"]


def analyze_appendix_forecast_ranking_values():
    values = pd.read_csv(LEVEL_SOURCE)
    _check_levels(values)
    main = pd.read_csv(SOURCE_DIR / "pairwise_refinement/group_summary.csv")
    pc = main[main.Metric.isin(["Preservation", "Refinement"])]
    return {"table": pd.concat([pc, values], ignore_index=True)}


def analyze_appendix_theme_value_losses():
    main = pd.read_csv(SOURCE_DIR / "theme_pairwise_loss_summary.csv")
    assert main.AnchorCondition.eq("same_theme_zeroed").all()
    values = pd.read_csv(LOSS_SOURCE)
    _check_losses(values, main)
    return {"table": main[main.Metric.isin(["P", "C"])].copy(), "values": values}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-anchors", action="store_true", help="Recompute cached shallow losses from raw local forecasts.")
    args = parser.parse_args()
    blocks = pd.read_csv(INPUTS / "date_blocks.csv")
    assert len(blocks) == 443 and blocks.DATE.is_unique
    base = np.load(INPUTS / "base_panel.npy", mmap_mode="r")
    monthly = pd.read_csv(SOURCE_DIR / "pairwise_refinement/monthly.csv")
    monthly = monthly[monthly["Primary Eligible"]].copy()
    assert not monthly.duplicated(["Model", "Width Schedule", "Depth", "DATE"]).any()
    full_input_levels(monthly, blocks, base)
    theme_losses(monthly, blocks, base, args.refresh_anchors)


if __name__ == "__main__":
    main()
