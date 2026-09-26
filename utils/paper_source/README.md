# Numerical companion

Open [exhibits.ipynb](exhibits.ipynb) to inspect the saved DataFrame tables and figures. Numerical inputs, caches and generated exports remain local and are ignored by Git. The repository retains the analysis/plotting code and two small predictor metadata files.

| Action | Command from repository root |
|---|---|
| List exhibits | `uv run python -m utils.paper_source --list` |
| Show a table | `uv run python -m utils.paper_source --asset FinalBaselinePerformance` |
| Draw a figure | `uv run python -m utils.paper_source --asset SpecificationDeltaSRHeatmaps` |
| Recompute one exhibit | `uv run python -m utils.paper_source --asset FinalBaselinePerformance --rebuild` |
| Verify local reference data | `uv run python -m utils.paper_source --all --verify` |
| Collect trained results | `uv run python -m utils.paper_source.collect --output tmp/replication_inputs` |

Rerunning notebook cells and preview commands requires local numerical caches; rebuilding also requires the corresponding input data. Verification recomputes estimates and compares them with locally available reference caches. A clean checkout contains saved notebook outputs but does not contain the underlying portfolio panels or numerical result caches. Use a new empty output folder when collecting your own experiment. The collector expects configuration prefixes such as `resnetps4d13`, as used by the experiment launcher; inspect the notebook's `quick_start/econ/` outputs directly. Set `RESASSETPRICING_SOURCE_DIR` and `RESASSETPRICING_RESULTS_DIR` to select its input and output folders.

| Exhibits | Required inputs |
|---|---|
| Portfolio performance / inference | Monthly portfolios, brief results, FF5, calibrated R2 |
| Transaction costs | Monthly portfolios and turnover |
| Decile profiles | Decile statistics and portfolio series |
| Preservation / refinement | Pairwise group/depth summaries |
| Theme effects | Theme-zeroed forecast summaries |
| Ranking-value appendix | Shallow/deep value summaries |
| Collapse diagnostic | Forecast/activation summaries |

Upstream routines are in `analysis/`: `forecast_level_r2`, `supplementary_experiments`, `export_conditional_forecast_inputs`, `forecast_refinement`, `theme_refinement`, and `ranking_value_appendix`. Their `--help` lists available inputs/options. Rebuilding stock-level diagnostics requires your licensed inputs and completed forecasts; it does not retrain models. Use `data/predictor_themes.csv` (columns `characteristic`, `cluster`) as `source_data/cluster_labels.csv` for the reference theme mapping.

P is unscaled; C is percent; V is monthly basis points over stock pairs. V is not a portfolio return. Monthly portfolio series use realization months. Exhibit notes specify eligible samples and inference. Details of training and resource budgets are in [the main README](../../readme.md).

Historical reference profiles retain the paper snapshot. Their cached FF5 alphas can use an older convention. New economic evaluations align factors with realized-return months and do not subtract RF again from a long-short return; recompute these alphas before comparing runs.
