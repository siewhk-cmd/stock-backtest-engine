from __future__ import annotations
import pandas as pd
from indicators import sma, roc


def prepare_benchmark(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    x = df[["Date", "Close"]].copy()
    for n in [20, 50, 100, 200]:
        x[f"{ticker}_MA{n}"] = sma(x["Close"], n)
    x[f"{ticker}_RET10"] = roc(x["Close"], 10)
    x[f"{ticker}_RET20"] = roc(x["Close"], 20)
    return x.rename(columns={"Close": f"{ticker}_Close"})


def merge_benchmarks(df: pd.DataFrame, benchmark_frames: dict[str, pd.DataFrame]):
    out = df.copy()
    for ticker, bdf in benchmark_frames.items():
        out = out.merge(prepare_benchmark(bdf, ticker), on="Date", how="left")
    return out
