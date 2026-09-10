from __future__ import annotations

from pathlib import Path
from typing import Iterable, Dict, Tuple

import pandas as pd

try:
    import streamlit as st
    cache_data = st.cache_data
except ImportError:
    # Allows non-Streamlit unit tests/imports.
    # In Streamlit Cloud, streamlit is installed from requirements.txt,
    # so the real @st.cache_data decorator is used.
    def cache_data(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


REQUIRED_COLUMNS = {"Date", "Open", "High", "Low", "Close"}
BENCHMARKS = {"SPY", "QQQ", "IWM", "GLD"}
KNOWN_ETFS = {"SPY", "QQQ", "IWM", "GLD", "ITA"}


def discover_tickers(data_dir: str | Path) -> list[str]:
    data_dir = Path(data_dir).expanduser()
    return sorted(p.stem.upper() for p in data_dir.glob("*.parquet"))


def classify_ticker(ticker: str) -> str:
    return "ETF" if ticker.upper() in KNOWN_ETFS else "Stock"


@cache_data(show_spinner=False)
def _load_ticker_cached(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    df = pd.read_parquet(path).copy()

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"{path.name}: missing required columns {sorted(missing)}"
        )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    numeric_cols = [
        "Open", "High", "Low", "Close", "Adj Close",
        "Volume", "Dividends", "Stock Splits"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=["Date", "Open", "High", "Low", "Close"])
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")
        .reset_index(drop=True)
    )
    return df


def load_ticker(path: str | Path) -> pd.DataFrame:
    """
    Load one ticker with Streamlit caching.

    Return a copy so downstream indicator calculations cannot accidentally
    mutate Streamlit's cached DataFrame.
    """
    path = Path(path).expanduser().resolve()
    return _load_ticker_cached(str(path)).copy()


def load_selected(
    data_dir: str | Path,
    tickers: Iterable[str],
    start=None,
    end=None,
) -> Tuple[Dict[str, pd.DataFrame], list[dict]]:
    data_dir = Path(data_dir).expanduser()
    out: Dict[str, pd.DataFrame] = {}
    errors: list[dict] = []

    for ticker in tickers:
        path = data_dir / f"{ticker}.parquet"
        try:
            df = load_ticker(path)

            if start is not None:
                df = df[df["Date"] >= pd.Timestamp(start)]
            if end is not None:
                df = df[df["Date"] <= pd.Timestamp(end)]

            if df.empty:
                raise ValueError("no rows in selected date range")

            out[ticker] = df.reset_index(drop=True)

        except Exception as exc:
            errors.append({"Ticker": ticker, "Error": str(exc)})

    return out, errors


@cache_data(show_spinner=False)
def _date_range_cached(data_dir_str: str, tickers_tuple: tuple[str, ...]):
    data_dir = Path(data_dir_str)
    mins, maxs = [], []

    for ticker in tickers_tuple:
        try:
            df = pd.read_parquet(
                data_dir / f"{ticker}.parquet",
                columns=["Date"],
            )
            d = pd.to_datetime(df["Date"], errors="coerce").dropna()
            if not d.empty:
                mins.append(d.min())
                maxs.append(d.max())
        except Exception:
            pass

    if not mins:
        return None, None
    return min(mins).date(), max(maxs).date()


def available_date_range(data_dir: str | Path, tickers: Iterable[str]):
    data_dir = Path(data_dir).expanduser().resolve()
    tickers_tuple = tuple(sorted(set(str(t).upper() for t in tickers)))
    return _date_range_cached(str(data_dir), tickers_tuple)


def validate_frame(ticker: str, df: pd.DataFrame) -> list[dict]:
    issues = []

    if df["Date"].duplicated().any():
        issues.append({"Ticker": ticker, "Issue": "Duplicate dates"})

    for col in ["Open", "High", "Low", "Close"]:
        bad = int(df[col].isna().sum())
        if bad:
            issues.append(
                {"Ticker": ticker, "Issue": f"{bad} missing {col} values"}
            )

        nonpos = int((df[col] <= 0).sum())
        if nonpos:
            issues.append(
                {"Ticker": ticker, "Issue": f"{nonpos} non-positive {col} values"}
            )

    if "Volume" in df.columns and df["Volume"].isna().any():
        issues.append({"Ticker": ticker, "Issue": "Missing volume values"})

    return issues
