from __future__ import annotations
import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def stochastic(df: pd.DataFrame, lookback=14, k_smooth=3, d_smooth=3):
    low_n = df["Low"].rolling(lookback, min_periods=lookback).min()
    high_n = df["High"].rolling(lookback, min_periods=lookback).max()
    raw_k = 100 * (df["Close"] - low_n) / (high_n - low_n).replace(0, np.nan)
    k = raw_k.rolling(k_smooth, min_periods=k_smooth).mean()
    d = k.rolling(d_smooth, min_periods=d_smooth).mean()
    return raw_k, k, d


def rsi(close: pd.Series, n=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def williams_r(df: pd.DataFrame, n=14):
    hh = df["High"].rolling(n, min_periods=n).max()
    ll = df["Low"].rolling(n, min_periods=n).min()
    return -100 * (hh - df["Close"]) / (hh - ll).replace(0, np.nan)


def cci(df: pd.DataFrame, n=20):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    ma = tp.rolling(n, min_periods=n).mean()
    mad = tp.rolling(n, min_periods=n).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    return (tp - ma) / (0.015 * mad.replace(0, np.nan))


def true_range(df: pd.DataFrame):
    pc = df["Close"].shift(1)
    return pd.concat([
        df["High"] - df["Low"],
        (df["High"] - pc).abs(),
        (df["Low"] - pc).abs(),
    ], axis=1).max(axis=1)


def atr(df: pd.DataFrame, n=14):
    return true_range(df).ewm(alpha=1/n, adjust=False, min_periods=n).mean()


def macd(close: pd.Series, fast=12, slow=26, signal=9):
    m = ema(close, fast) - ema(close, slow)
    sig = ema(m, signal)
    return m, sig, m - sig


def bollinger(close: pd.Series, n=20, std_mult=2.0):
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std()
    upper = mid + std_mult * sd
    lower = mid - std_mult * sd
    bandwidth = (upper - lower) / mid.replace(0, np.nan) * 100
    return mid, upper, lower, bandwidth


def roc(close: pd.Series, n=20):
    return close.pct_change(n) * 100


def adx(df: pd.DataFrame, n=14):
    up = df["High"].diff()
    down = -df["Low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr = true_range(df)
    atr_n = tr.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1/n, adjust=False, min_periods=n).mean() / atr_n.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1/n, adjust=False, min_periods=n).mean() / atr_n.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_v = dx.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    return adx_v, plus_di, minus_di
