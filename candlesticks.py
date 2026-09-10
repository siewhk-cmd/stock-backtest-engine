from __future__ import annotations
import numpy as np
import pandas as pd


def _parts(df):
    body = (df["Close"] - df["Open"]).abs()
    rng = (df["High"] - df["Low"]).replace(0, np.nan)
    upper = df["High"] - df[["Open", "Close"]].max(axis=1)
    lower = df[["Open", "Close"]].min(axis=1) - df["Low"]
    bullish = df["Close"] > df["Open"]
    bearish = df["Close"] < df["Open"]
    return body, rng, upper, lower, bullish, bearish


def _split_safe(df, signal):
    if "Stock Splits" in df.columns:
        split = df["Stock Splits"].fillna(0).ne(0)
        signal &= ~(split | split.shift(1).fillna(False))
    return signal.fillna(False)


def bullish_engulfing(df):
    po, pc = df["Open"].shift(1), df["Close"].shift(1)
    signal = (pc < po) & (df["Close"] > df["Open"]) & (df["Open"] <= pc) & (df["Close"] >= po)
    return _split_safe(df, signal)


def bearish_engulfing(df):
    po, pc = df["Open"].shift(1), df["Close"].shift(1)
    signal = (pc > po) & (df["Close"] < df["Open"]) & (df["Open"] >= pc) & (df["Close"] <= po)
    return _split_safe(df, signal)


def hammer(df):
    body, rng, upper, lower, _, _ = _parts(df)
    return ((lower >= 2*body) & (upper <= body) & ((body/rng) <= 0.4)).fillna(False)


def inverted_hammer(df):
    body, rng, upper, lower, _, _ = _parts(df)
    return ((upper >= 2*body) & (lower <= body) & ((body/rng) <= 0.4)).fillna(False)


def shooting_star(df): return (inverted_hammer(df) & (df["Close"] < df["Open"])).fillna(False)
def hanging_man(df): return (hammer(df) & (df["Close"] < df["Open"])).fillna(False)


def doji(df):
    body, rng, _, _, _, _ = _parts(df)
    return ((body / rng) <= 0.10).fillna(False)


def spinning_top(df):
    body, rng, upper, lower, _, _ = _parts(df)
    return (((body/rng) <= 0.30) & (upper > body*0.5) & (lower > body*0.5)).fillna(False)


def inside_bar(df): return ((df["High"] < df["High"].shift(1)) & (df["Low"] > df["Low"].shift(1))).fillna(False)
def outside_bar(df): return ((df["High"] > df["High"].shift(1)) & (df["Low"] < df["Low"].shift(1))).fillna(False)


def bullish_harami(df):
    po, pc = df["Open"].shift(1), df["Close"].shift(1)
    return ((pc < po) & (df["Close"] > df["Open"]) & (df["Open"] >= pc) & (df["Close"] <= po)).fillna(False)


def bearish_harami(df):
    po, pc = df["Open"].shift(1), df["Close"].shift(1)
    return ((pc > po) & (df["Close"] < df["Open"]) & (df["Open"] <= pc) & (df["Close"] >= po)).fillna(False)


def piercing_pattern(df):
    po, pc = df["Open"].shift(1), df["Close"].shift(1)
    midpoint = (po + pc) / 2
    return ((pc < po) & (df["Close"] > df["Open"]) & (df["Open"] <= pc) & (df["Close"] > midpoint) & (df["Close"] < po)).fillna(False)


def dark_cloud_cover(df):
    po, pc = df["Open"].shift(1), df["Close"].shift(1)
    midpoint = (po + pc) / 2
    return ((pc > po) & (df["Close"] < df["Open"]) & (df["Open"] >= pc) & (df["Close"] < midpoint) & (df["Close"] > po)).fillna(False)


def morning_star(df):
    body = (df["Close"] - df["Open"]).abs()
    avg_body = body.rolling(20, min_periods=5).mean()
    midpoint_first = (df["Open"].shift(2) + df["Close"].shift(2)) / 2
    return ((df["Close"].shift(2) < df["Open"].shift(2)) & (body.shift(2) > avg_body.shift(2)) &
            (body.shift(1) < avg_body.shift(1)*0.6) & (df["Close"] > df["Open"]) & (df["Close"] > midpoint_first)).fillna(False)


def evening_star(df):
    body = (df["Close"] - df["Open"]).abs()
    avg_body = body.rolling(20, min_periods=5).mean()
    midpoint_first = (df["Open"].shift(2) + df["Close"].shift(2)) / 2
    return ((df["Close"].shift(2) > df["Open"].shift(2)) & (body.shift(2) > avg_body.shift(2)) &
            (body.shift(1) < avg_body.shift(1)*0.6) & (df["Close"] < df["Open"]) & (df["Close"] < midpoint_first)).fillna(False)


def three_white_soldiers(df):
    bull = df["Close"] > df["Open"]
    higher = (df["Close"] > df["Close"].shift(1)) & (df["Close"].shift(1) > df["Close"].shift(2))
    return (bull & bull.shift(1) & bull.shift(2) & higher).fillna(False)


def three_black_crows(df):
    bear = df["Close"] < df["Open"]
    lower = (df["Close"] < df["Close"].shift(1)) & (df["Close"].shift(1) < df["Close"].shift(2))
    return (bear & bear.shift(1) & bear.shift(2) & lower).fillna(False)


def tweezer_bottom(df, tolerance=0.002):
    rel = (df["Low"] - df["Low"].shift(1)).abs() / df["Close"].replace(0, np.nan)
    return ((df["Close"].shift(1) < df["Open"].shift(1)) & (df["Close"] > df["Open"]) & (rel <= tolerance)).fillna(False)


def tweezer_top(df, tolerance=0.002):
    rel = (df["High"] - df["High"].shift(1)).abs() / df["Close"].replace(0, np.nan)
    return ((df["Close"].shift(1) > df["Open"].shift(1)) & (df["Close"] < df["Open"]) & (rel <= tolerance)).fillna(False)


def dragonfly_doji(df):
    body, rng, upper, lower, _, _ = _parts(df)
    return (((body/rng) <= 0.10) & (upper/rng <= 0.10) & (lower/rng >= 0.60)).fillna(False)


def gravestone_doji(df):
    body, rng, upper, lower, _, _ = _parts(df)
    return (((body/rng) <= 0.10) & (lower/rng <= 0.10) & (upper/rng >= 0.60)).fillna(False)

PATTERNS = {
    "Bullish Engulfing": bullish_engulfing,
    "Bearish Engulfing": bearish_engulfing,
    "Hammer": hammer,
    "Inverted Hammer": inverted_hammer,
    "Shooting Star": shooting_star,
    "Hanging Man": hanging_man,
    "Morning Star": morning_star,
    "Evening Star": evening_star,
    "Piercing Pattern": piercing_pattern,
    "Dark Cloud Cover": dark_cloud_cover,
    "Three White Soldiers": three_white_soldiers,
    "Three Black Crows": three_black_crows,
    "Bullish Harami": bullish_harami,
    "Bearish Harami": bearish_harami,
    "Tweezer Bottom": tweezer_bottom,
    "Tweezer Top": tweezer_top,
    "Dragonfly Doji": dragonfly_doji,
    "Gravestone Doji": gravestone_doji,
    "Doji": doji,
    "Spinning Top": spinning_top,
    "Inside Bar": inside_bar,
    "Outside Bar": outside_bar,
}


def apply_patterns(df: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    out = df.copy()
    for name in names:
        fn = PATTERNS.get(name)
        if fn is not None:
            out[f"PATTERN::{name}"] = fn(out)
    return out
