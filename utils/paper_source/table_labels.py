"""Pure label transformations shared by analysis and rendering."""
import numpy as np
import pandas as pd
from utils.paper_source.table_spec import WIDTH_SCHEDULE_LABELS

def _sr_stars(p_value: float) -> str:
    if not np.isfinite(p_value):
        return ""
    if p_value <= 0.01:
        return "***"
    if p_value <= 0.05:
        return "**"
    if p_value <= 0.10:
        return "*"
    return ""


def _tstat_stars(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    t_stat = float(value)
    if not np.isfinite(t_stat):
        return ""
    abs_t = abs(t_stat)
    if abs_t >= 2.576:
        return "***"
    if abs_t >= 1.960:
        return "**"
    if abs_t >= 1.645:
        return "*"
    return ""


def _width_schedule_label(seed: object) -> str:
    try:
        value = int(seed)
    except (TypeError, ValueError):
        return str(seed)
    return WIDTH_SCHEDULE_LABELS.get(value, f"S{value}")
