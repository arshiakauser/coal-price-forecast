"""Loading, cleaning and splitting the price data.

Nothing in this module scales or otherwise learns statistics from the data,
so it is safe to call before the train/test split.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import PRICE_COLUMNS


def load_raw(source) -> pd.DataFrame:
    """Read a raw CSV/XLSX (path or uploaded file object) into a clean DataFrame.

    Handles the layout of the original export: the first column holds dates
    (header may be blank / "Unnamed: 0"), and the first data row may repeat
    the header words ("Date", "Price", ...). Rows whose date cannot be parsed
    are dropped; price columns are coerced to numeric.
    """
    name = str(getattr(source, "name", source)).lower()
    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(source)
    else:
        df = pd.read_csv(source)

    date_col = df.columns[0]
    # strip suffixes such as "2020-04-02 [GMT]"
    dates = df[date_col].astype(str).str.replace(r"\[.*\]", "", regex=True).str.strip()
    df.index = pd.to_datetime(dates, errors="coerce", format="mixed")
    df.index.name = "Date"
    df = df.drop(columns=[date_col])
    df = df[df.index.notna()]  # drops the stray "Date/Price" header row

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df.dropna(axis=1, how="all")


def clean_series(s: pd.Series) -> pd.Series:
    """Fill gaps using only past values (forward fill) and drop leading NaNs."""
    return s.ffill().dropna().astype(float)


def select_columns(df: pd.DataFrame, requested: list[str] | None = None) -> list[str]:
    """Pick the price columns to model."""
    if requested:
        missing = [c for c in requested if c not in df.columns]
        if missing:
            raise KeyError(f"Columns not found in data: {missing}")
        return requested
    present = [c for c in PRICE_COLUMNS if c in df.columns]
    return present or list(df.select_dtypes("number").columns)


def ensure_dir(path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
