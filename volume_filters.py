from __future__ import annotations
import pandas as pd


def avg_volume(df, n=20):
    return df["Volume"].rolling(n, min_periods=n).mean()


def relative_volume(df, n=20):
    av = avg_volume(df, n)
    return df["Volume"] / av.replace(0, pd.NA)


def volume_increasing(df, n=3):
    up = df["Volume"] > df["Volume"].shift(1)
    return up.rolling(n).sum().eq(n)
