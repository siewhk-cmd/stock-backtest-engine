from __future__ import annotations
import numpy as np
import pandas as pd


def summarize_signal_study(signals: pd.DataFrame, windows=(1,3,5,10,20)) -> pd.DataFrame:
    rows = []
    if signals.empty:
        return pd.DataFrame()
    for w in windows:
        c = f"{w}D_ReturnPct"
        if c not in signals.columns:
            continue
        v = signals[c].dropna()
        if v.empty:
            continue
        rows.append({
            "HoldingDays": w,
            "Signals": len(v),
            "WinRatePct": v.gt(0).mean()*100,
            "AvgReturnPct": v.mean(),
            "MedianReturnPct": v.median(),
            "BestReturnPct": v.max(),
            "WorstReturnPct": v.min(),
            "AvgMFEPct": signals.loc[v.index, f"{w}D_MFE_Pct"].mean(),
            "AvgMAEPct": signals.loc[v.index, f"{w}D_MAE_Pct"].mean(),
        })
    return pd.DataFrame(rows)


def summarize_trades(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {}
    r = trades["ReturnPct"].dropna()
    wins, losses = r[r > 0], r[r <= 0]
    gross_profit, gross_loss = wins.sum(), -losses.sum()
    return {
        "Trades": len(r),
        "WinRatePct": r.gt(0).mean()*100,
        "AvgReturnPct": r.mean(),
        "MedianReturnPct": r.median(),
        "AvgGainPct": wins.mean() if len(wins) else np.nan,
        "AvgLossPct": losses.mean() if len(losses) else np.nan,
        "ProfitFactor": gross_profit/gross_loss if gross_loss > 0 else np.inf,
        "BestTradePct": r.max(),
        "WorstTradePct": r.min(),
        "AvgMFEPct": trades["MFE_Pct"].mean(),
        "AvgMAEPct": trades["MAE_Pct"].mean(),
    }


def by_ticker(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows = []
    for t, g in trades.groupby("Ticker"):
        r = g["ReturnPct"]
        rows.append({
            "Ticker": t,
            "Trades": len(g),
            "WinRatePct": r.gt(0).mean()*100,
            "AvgReturnPct": r.mean(),
            "MedianReturnPct": r.median(),
            "BestTradePct": r.max(),
            "WorstTradePct": r.min(),
            "AvgMFEPct": g["MFE_Pct"].mean(),
            "AvgMAEPct": g["MAE_Pct"].mean(),
        })
    return pd.DataFrame(rows).sort_values(["Trades","Ticker"], ascending=[False,True])


# ---------- Short put metrics ----------

def summarize_short_puts(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {}
    n = len(trades)
    assigned = trades["AssignmentProxy"].fillna(False).astype(bool)
    touched = trades["StrikeTouched"].fillna(False).astype(bool)
    at_strike = trades["ExpiryState"].eq("At Strike")
    otm = trades["ExpiryState"].eq("OTM")
    recovered = trades["RecoveryClassification"].eq("Touched & Recovered OTM")
    never_touched = trades["RecoveryClassification"].eq("Never Touched Strike")
    touched_itm = trades["RecoveryClassification"].eq("Touched & Expired ITM")
    return {
        "ShortPutsTested": n,
        "ExpiredOTM_Count": int(otm.sum()),
        "ExpiredOTM_Pct": float(otm.mean() * 100),
        "AssignmentProxy_Count": int(assigned.sum()),
        "AssignmentProxy_Pct": float(assigned.mean() * 100),
        "AtStrike_Count": int(at_strike.sum()),
        "StrikeTouched_Count": int(touched.sum()),
        "StrikeTouched_Pct": float(touched.mean() * 100),
        "TouchedRecovered_Count": int(recovered.sum()),
        "TouchedRecovered_Pct": float(recovered.mean() * 100),
        "TouchedExpiredITM_Count": int(touched_itm.sum()),
        "NeverTouched_Count": int(never_touched.sum()),
        "NeverTouched_Pct": float(never_touched.mean() * 100),
        "AvgExpiryMarginPct": float(trades["ExpiryMarginPct"].mean()),
        "MedianExpiryMarginPct": float(trades["ExpiryMarginPct"].median()),
        "AvgUnderlyingReturnPct": float(trades["UnderlyingReturnPct"].mean()),
        "MedianUnderlyingReturnPct": float(trades["UnderlyingReturnPct"].median()),
        "AvgMAE_Pct": float(trades["MAE_Pct"].mean()),
        "AvgMFE_Pct": float(trades["MFE_Pct"].mean()),
        "AvgMaxStrikeBreachPct": float(trades["MaxStrikeBreachPct"].mean()),
        "WorstStrikeBreachPct": float(trades["MaxStrikeBreachPct"].min()),
    }


def _short_put_group_table(trades: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows = []
    for keys, g in trades.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        s = summarize_short_puts(g)
        row.update({
            "Trades": s["ShortPutsTested"],
            "ExpiredOTM_Pct": s["ExpiredOTM_Pct"],
            "AssignmentProxy_Pct": s["AssignmentProxy_Pct"],
            "StrikeTouch_Pct": s["StrikeTouched_Pct"],
            "TouchedRecovered_Pct": s["TouchedRecovered_Pct"],
            "AvgExpiryMarginPct": s["AvgExpiryMarginPct"],
            "AvgUnderlyingReturnPct": s["AvgUnderlyingReturnPct"],
            "AvgMAE_Pct": s["AvgMAE_Pct"],
            "AvgMFE_Pct": s["AvgMFE_Pct"],
            "AvgMaxStrikeBreachPct": s["AvgMaxStrikeBreachPct"],
            "WorstStrikeBreachPct": s["WorstStrikeBreachPct"],
        })
        rows.append(row)
    return pd.DataFrame(rows)


def short_put_by_otm_dte(trades: pd.DataFrame) -> pd.DataFrame:
    x = _short_put_group_table(trades, ["OTM_Pct", "TargetDTE_CalendarDays"])
    if x.empty:
        return x
    return x.sort_values(["TargetDTE_CalendarDays", "OTM_Pct"]).reset_index(drop=True)


def short_put_by_otm(trades: pd.DataFrame) -> pd.DataFrame:
    x = _short_put_group_table(trades, ["OTM_Pct"])
    return x.sort_values("OTM_Pct").reset_index(drop=True) if not x.empty else x


def short_put_by_dte(trades: pd.DataFrame) -> pd.DataFrame:
    x = _short_put_group_table(trades, ["TargetDTE_CalendarDays"])
    return x.sort_values("TargetDTE_CalendarDays").reset_index(drop=True) if not x.empty else x


def short_put_by_ticker(trades: pd.DataFrame) -> pd.DataFrame:
    x = _short_put_group_table(trades, ["Ticker"])
    if x.empty:
        return x
    return x.sort_values(["Trades", "Ticker"], ascending=[False, True]).reset_index(drop=True)


def short_put_recovery_analysis(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    counts = trades["RecoveryClassification"].value_counts(dropna=False)
    n = len(trades)
    rows = []
    for name, count in counts.items():
        rows.append({
            "RecoveryClassification": name,
            "Count": int(count),
            "PctOfTrades": float(count / n * 100),
        })
    return pd.DataFrame(rows).sort_values("Count", ascending=False).reset_index(drop=True)
