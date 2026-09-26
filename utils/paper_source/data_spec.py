"""Static source names and experimental grid; no file reads."""
import re
from utils.paper_source.paths import SOURCE_DIR

PREFIX_RE = re.compile(r"^(?P<stem>resnetps|nnps|resnets|nns)(?P<seed>\d+)d(?P<depth>\d+)$")

FAMILY_NAMES = {
    "resnetps": "ResNet+",
    "nnps": "NN+",
    "resnets": "ResNet",
    "nns": "NN",
}

MAIN_FAMILIES = ("ResNet+", "NN+")

SUPPLEMENTARY_FAMILIES = ("ResNet", "NN")

REPORT_FAMILIES = MAIN_FAMILIES + SUPPLEMENTARY_FAMILIES

PUBLIC_WIDTH_SCHEDULES = (1, 2, 3, 4)

PLANNED_MAX_DEPTH = 20

PRIMARY_BENCHMARK_TEMPLATE = 4

PRIMARY_BENCHMARK_DEPTH = 10

PRIMARY_PAIRED_TEMPLATE = 3

PRIMARY_PAIRED_DEPTH = PLANNED_MAX_DEPTH

BRIEF_PANEL = SOURCE_DIR / "brief_results_panel.csv"

BRIEF_ALL = SOURCE_DIR / "brief_results_all.csv"

DECILE_PANEL = SOURCE_DIR / "decile_returns"

TURNOVER_PANEL = SOURCE_DIR / "turnover_panel.csv.gz"

COVERAGE_SUMMARY = SOURCE_DIR / "coverage_summary.csv"

FALLBACK_MONTH_COLUMNS = (
    "Seeded Tie-Split Decile Fallback Months",
    "Constant-Predict Decile Fallback Months",
)

CONSTANT_PREDICT_FALLBACK_COLUMN = "Constant-Predict Decile Fallback Months"
