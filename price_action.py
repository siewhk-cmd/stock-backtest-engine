from __future__ import annotations
import pandas as pd


def previous_return(df, days: int):
    return df["Close"].pct_change(days) * 100


def consecutive_up(df, n: int):
    up = df["Close"] > df["Close"].shift(1)
    return up.rolling(n).sum().eq(n)


def consecutive_down(df, n: int):
    dn = df["Close"] < df["Close"].shift(1)
    return dn.rolling(n).sum().eq(n)


def gap_pct(df):
    return (df["Open"] / df["Close"].shift(1) - 1) * 100


def distance_from_low(df, n: int):
    low_n = df["Low"].rolling(n, min_periods=n).min()
    return (df["Close"] / low_n - 1) * 100


def distance_from_high(df, n: int):
    high_n = df["High"].rolling(n, min_periods=n).max()
    return (df["Close"] / high_n - 1) * 100


def breakout(df, n: int):
    prev_high = df["High"].shift(1).rolling(n, min_periods=n).max()
    return df["Close"] > prev_high


def breakdown(df, n: int):
    prev_low = df["Low"].shift(1).rolling(n, min_periods=n).min()
    return df["Close"] < prev_low
