"""Shared exhibit labels and presentation constants; no data reads."""
from utils.paper_source.paths import SOURCE_DIR

REPORT_MODEL_ORDER = ("NN", "ResNet", "NN+", "ResNet+")

VW_SR = "Value Weighted Long Short Sharpe Ratio"

EW_SR = "Equal Weighted Long Short Sharpe Ratio"

VW_ALPHA = "Value Weighted LS FF5 Alpha/%"

VW_ALPHA_T = "Value Weighted LS FF5 t-stat"

VW_TURNOVER = "Value Weighted LS Turnover/%"

EW_TURNOVER = "Equal Weighted LS Turnover/%"

OOS_R2 = "Out-of-Sample R2"

TABLE_R2_OOS = "$R^2_{OOS}$"

TABLE_R2_SCALE = "Scale"

LW_T_COL = "$t_{\\mathrm{LW}}$"

FORECAST_LEVEL_R2_SOURCE = SOURCE_DIR / "forecast_level_r2_variants.csv"

DECILE_PORTFOLIO_STATISTICS_SOURCE = SOURCE_DIR / "decile_portfolio_statistics.csv"

FORECAST_LEVEL_R2_COLUMN = "cs_demeaned_trim_2p5_each_tail_scaled_r2oos"

FORECAST_LEVEL_SCALE_COLUMN = "cs_demeaned_trim_2p5_each_tail_scale"

FF5_FACTOR_PATHS = (
    SOURCE_DIR / "ff5_factors_monthly.csv",
)

PREDICTOR_LIST_SOURCE = SOURCE_DIR / "predictor_list.csv"

NBER_BUSINESS_CYCLE_PEAK_TROUGH = (
    ("1990-07", "1991-03"),
    ("2001-03", "2001-11"),
    ("2007-12", "2009-06"),
    ("2020-02", "2020-04"),
)

MARKET_STATE_INFERENCE_MIN_MONTHS = 13

INITIAL_LONG_SHORT_GROSS_TURNOVER = 2.0

WIDTH_SCHEDULE_LABELS = {
    1: "S1=[32]",
    2: "S2=[32,16]",
    3: "S3=[32,16,8]",
    4: "S4=[32,16,8,4]",
}

RAW_RESULT_COLUMNS = [
    "Width",
    "Depth",
    "Model",
    "Spec",
    "Months",
    "$R^2_{OOS}$",
    "Scale",
    "VW SR",
    "FF5 $\\alpha$",
    "$t(\\alpha)$",
    "MDD",
    "Max loss",
    "Turnover",
    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
    "D6",
    "D7",
    "D8",
    "D9",
    "D10",
    "H-L",
]

DEPTH_PROFILE_COLUMNS = [
    "Width",
    "Depth",
    "Model",
    "Spec",
    "$R^2_{OOS}$",
    "Scale",
    "VW SR",
    "FF5 $\\alpha$",
    "$t(\\alpha)$",
    "Turnover",
    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
    "D6",
    "D7",
    "D8",
    "D9",
    "D10",
    "H-L",
]

DEPTH_PROFILE_HEADERS = {
    "$R^2_{OOS}$": r"$R^2_{\mathrm{OOS}}$",
    "VW SR": r"VW~SR",
    "FF5 $\\alpha$": r"FF5~$\alpha$",
    "H-L": r"H$-$L",
}



PAIR_METRIC_COLUMNS = [
    "Value Weighted Long Short Sharpe Ratio",
    "Out-of-Sample R2",
    "Value Weighted LS FF5 Alpha/%",
    "Value Weighted LS FF5 t-stat",
    "Value Weighted Long Short Maximum Drawdown",
    "Value Weighted Long Short Maximum One Month Loss",
    "Value Weighted LS Turnover/%",
    "Trainable Parameters",
]

PREDICTOR_DESCRIPTION_LABELS = {
    "Change PPE and inventory": "Change in PPE and inventory",
    "Highest 5 days of return": "Highest 5 daily returns",
    "Abnormal corporate investment": "Abnormal investment",
    "Book-to-market enterprise value": "Book to enterprise value",
    "Change gross margin minus change sales": "Gross margin change minus sales change",
    "Change in current operating assets": "Change in current operating assets",
    "Change in current operating liabilities": "Change in current operating liabilities",
    "Change in current operating working capital": "Change in current working capital",
    "Change in financial liabilities": "Change in financial liabilities",
    "Change in long-term investments": "Change in long-term investments",
    "Change in long-term net operating assets": "Change in long-term operating assets",
    "Change in net financial assets": "Change in net financial assets",
    "Change in net noncurrent operating assets": "Change in net noncurrent assets",
    "Change in net operating assets": "Change in net operating assets",
    "Change in noncurrent operating assets": "Change in noncurrent assets",
    "Change in noncurrent operating liabilities": "Change in noncurrent liabilities",
    "Change in operating cash flow to assets": "Change in cash flow to assets",
    "Change in quarterly return on assets": "Change in quarterly ROA",
    "Change in quarterly return on equity": "Change in quarterly ROE",
    "Change in short-term investments": "Change in short-term investments",
    "Change sales minus change inventory": "Sales change minus inventory change",
    "Change sales minus change receivables": "Sales change minus receivables change",
    "Change sales minus change SG&A": "Sales change minus SG&A change",
    "Number of consecutive quarters with earnings increases": "Consecutive earnings increases",
    "Current price to high price over last year": "Price to 12-month high",
    "EBITDA-to-market enterprise value": "EBITDA to enterprise value",
    "Cash-based operating profits-to-book assets": "Cash operating profit to assets",
    "Cash-based operating profits-to-lagged book assets": "Cash operating profit to lagged assets",
    "Coefficient of variation for dollar trading volume": "Dollar volume variation",
    "Coefficient of variation for share turnover": "Turnover variation",
    "Frazzini-Pedersen market beta": "Frazzini-Pedersen beta",
    "Highest 5 days of return scaled by volatility": "Top five returns over volatility",
    "Idiosyncratic skewness from the CAPM": "CAPM idiosyncratic skewness",
    "Idiosyncratic skewness from the Fama-French 3-factor model": "FF3 idiosyncratic skewness",
    "Idiosyncratic skewness from the q-factor model": "q-factor idiosyncratic skewness",
    "Idiosyncratic volatility from the CAPM (21 days)": "CAPM idiosyncratic volatility, 21 days",
    "Idiosyncratic volatility from the CAPM (252 days)": "CAPM idiosyncratic volatility, 252 days",
    "Idiosyncratic volatility from the Fama-French 3-factor model": "FF3 idiosyncratic volatility",
    "Idiosyncratic volatility from the q-factor model": "q-factor idiosyncratic volatility",
    "Number of zero trades with turnover as tiebreaker (1 month)": "Zero trade days, 1 month",
    "Number of zero trades with turnover as tiebreaker (6 months)": "Zero trade days, 6 months",
    "Number of zero trades with turnover as tiebreaker (12 months)": "Zero trade days, 12 months",
    "Operating cash flow-to-market": "Cash flow to market",
    "Operating profits-to-book assets": "Operating profit to assets",
    "Operating profits-to-book equity": "Operating profit to equity",
    "Operating profits-to-lagged book assets": "Operating profit to lagged assets",
    "Operating profits-to-lagged book equity": "Operating profit to lagged equity",
    "Price momentum t-12 to t-1": "12-month price momentum",
    "Quarterly return on assets": "Quarterly ROA",
    "Quarterly return on equity": "Quarterly ROE",
    "Quality minus Junk: Composite": "Quality minus junk composite",
    "Quality minus Junk: Growth": "Quality minus junk growth",
    "Quality minus Junk: Profitability": "Quality minus junk profitability",
    "Quality minus Junk: Safety": "Quality minus junk safety",
    "Residual momentum t-12 to t-1": "12-month residual momentum",
    "Residual momentum t-6 to t-1": "6-month residual momentum",
    "Return on net operating assets": "Return on operating assets",
    "R&D capital-to-book assets": "R&D capital to assets",
    "Taxable income-to-book income": "Taxable to book income",
    "The high-low bid-ask spread": "High-low bid-ask spread",
}

BASELINE_DEPTH_BANDS = (
    ("Shallow", 1, 5),
    ("Medium", 6, 10),
    ("Deep", 11, 20),
)

BASELINE_TABLE_COLUMNS = (
    "VW Mean",
    "VW Vol.",
    "VW SR",
    "$\\Delta$ VW SR",
    "EW SR",
    TABLE_R2_OOS,
    "FF5 $\\alpha$",
    "Turnover",
)

SR_BOOTSTRAP_SEEDS = {
    ("ResNet+", "NN"): {
        "Shallow": 20260616,
        "Medium": 20260617,
        "Deep": 20260618,
        "All": 20260619,
    },
    ("ResNet", "NN"): {
        "Shallow": 20260510,
        "Medium": 20260511,
        "Deep": 20260507,
        "All": 20260506,
    },
    ("ResNet+", "NN+"): {
        "Shallow": 20260512,
        "Medium": 20260513,
        "Deep": 20260509,
        "All": 20260508,
    },
}

DEPTH_CONTRAST_BOOTSTRAP_SEEDS = {
    "ResNet": 20260920,
    "NN": 20260921,
    "ResNet+": 20260922,
    "NN+": 20260923,
}

DEPTH_BANDS = (
    ("Shallow", 1, 5),
    ("Medium", 6, 10),
    ("Deep", 11, 20),
)

DEPTH_GROUP_NOTE = (
    "Shallow, medium, and deep groups cover hidden depths 1--5, 6--10, and 11--20."
)

PAIRED_SPEC_AVERAGE_NOTE = (
    "Reported averages weight completed paired width and depth specifications equally."
)
