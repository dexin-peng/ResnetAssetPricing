"""Portable inputs and results for the research companion."""
from pathlib import Path
import os
PAPER_SOURCE_DIR = Path(__file__).resolve().parent
REPORT_DIR = PAPER_SOURCE_DIR
REPO_ROOT = PAPER_SOURCE_DIR.parents[1]
SOURCE_DIR = Path(os.environ.get("RESASSETPRICING_SOURCE_DIR", PAPER_SOURCE_DIR / "data")).resolve()
RESULTS_DIR = Path(os.environ.get("RESASSETPRICING_RESULTS_DIR", PAPER_SOURCE_DIR / "results")).resolve()
FIG_DIR = RESULTS_DIR / "figures"
def ensure_asset_dirs():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
