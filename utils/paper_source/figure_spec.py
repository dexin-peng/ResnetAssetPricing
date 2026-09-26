"""Shared figure labels and display constants, without empirical calculations."""

VW_SR = "Value Weighted Long Short Sharpe Ratio"

RECESSIONS = [
    ("1990-07-01", "1991-03-01"),
    ("2001-03-01", "2001-11-01"),
    ("2007-12-01", "2009-06-01"),
    ("2020-02-01", "2020-04-01"),
]

PARAMETER_SCALE_EXTERNAL_RECORDS = [
    {
        "label": "CAPM",
        "domain": "Asset Pricing",
        "low": 2,
        "high": 2,
        "year": "1964",
    },
    {
        "label": "Fama-French three factor",
        "domain": "Asset Pricing",
        "low": 4,
        "high": 4,
        "year": "1993",
    },
    {
        "label": "Carhart four factor",
        "domain": "Asset Pricing",
        "low": 5,
        "high": 5,
        "year": "1997",
    },
    {
        "label": "Barra USE4",
        "domain": "Asset Pricing",
        "low": 72,
        "high": 72,
        "year": "2013",
    },
    {
        "label": "Factor zoo",
        "domain": "Asset Pricing",
        "low": 316,
        "high": 316,
        "year": "2016",
    },
    {
        "label": "BERT base to large",
        "domain": "External ML",
        "low": 110_000_000,
        "high": 340_000_000,
        "year": "2019",
    },
    {
        "label": "Whisper family",
        "domain": "External ML",
        "low": 39_000_000,
        "high": 1_550_000_000,
        "year": "2022",
    },
    {
        "label": "LLaMA family",
        "domain": "External ML",
        "low": 6_700_000_000,
        "high": 65_200_000_000,
        "year": "2023",
    },
    {
        "label": "Chinchilla",
        "domain": "External ML",
        "low": 70_000_000_000,
        "high": 70_000_000_000,
        "year": "2022",
    },
    {
        "label": "GPT-3",
        "domain": "External ML",
        "low": 175_000_000_000,
        "high": 175_000_000_000,
        "year": "2020",
    },
    {
        "label": "PaLM",
        "domain": "External ML",
        "low": 540_000_000_000,
        "high": 540_000_000_000,
        "year": "2022",
    },
]
