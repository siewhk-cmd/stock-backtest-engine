from __future__ import annotations

import math
from typing import Iterable
import numpy as np
import pandas as pd

from backtest_engine import build_signal


PRICE_EQUAL_TOLERANCE = 0.005  # half a cent for "At Strike"


def round_strike_down(strike: float, rule: str) -> float:
    """Round a synthetic strike down using the selected increment."""
    increments = {
        "No rounding": None,
        "Nearest $0.50 down": 0.50,
        "Nearest $1 down": 1.00,
        "Nearest $2.50 down": 2.50,
        "Nearest $5 down": 5.00,
    }
    inc = increments.get(rule)
    if inc is None:
        return float(strike)
    if strike <= 0:
        return float("nan")
    return math.floor((strike + 1e-12) / inc) * inc


def _classify_recovery(strike_touched: bool, expiry_state: str) -> str:
    if expiry_state == "At Strike":
        return "At Strike"
    if expiry_state == "ITM / Assignment Proxy":
        if strike_touched:
            return "Touched & Expired ITM"
        return "DATA WARNING: Expired ITM without recorded touch"
    if strike_touched:
        return "Touched & Recovered OTM"
    return "Never Touched Strike"


def _signal_snapshot(work: pd.DataFrame, i: int) -> dict:
    """Capture useful signal-day values when they exist."""
    row = {}
    common = [
        "STOCH_K", "STOCH_D", "RSI", "WILLR", "CCI", "MACD",
        "MACD_SIGNAL", "ADX", "PLUS_DI", "MINUS_DI", "ATR_PCT",
        "ROC", "REL_VOLUME", "MA20", "MA50", "MA100", "MA200",
        "PA_RETURN", "GAP_PCT", "DIST_LOW_PCT", "DIST_HIGH_PCT",
        "DIST_MA_PCT",
    ]
    for col in common:
        if col in work.columns:
            row[col] = work.at[i, col]

    pattern_cols = [c for c in work.columns if c.startswith("PATTERN::")]
    active = []
    for col in pattern_cols:
        try:
            if bool(work.at[i, col]):
                active.append(col.split("::", 1)[1])
        except Exception:
            pass
    if active:
        row["MatchedPatterns"] = " | ".join(active)

    # Preserve benchmark values that were merged into the signal frame.
    for col in work.columns:
        if col.startswith(("SPY_", "QQQ_", "IWM_", "GLD_")):
            row[col] = work.at[i, col]
    return row


def short_puts_for_ticker(
    ticker: str,
    df: pd.DataFrame,
    config: dict,
    benchmark_frames: dict[str, pd.DataFrame] | None,
    short_put_config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create synthetic short puts from signal dates.

    Assumptions:
    - Signal known after signal-day close.
    - Put sold at next trading day's Open.
    - Synthetic strike is an OTM percentage below entry underlying price.
    - DTE is calendar days.
    - If target expiry is weekend/holiday, expiry is last trading day
      on or before the target date.
    - If price history ends before target expiry, that combination is skipped.
    - Expiry Close < strike is an expiration ITM / assignment proxy only.
    """
    work, signal = build_signal(df, config, benchmark_frames)
    if work.empty:
        return pd.DataFrame(), work

    otm_levels = sorted({float(x) for x in short_put_config.get("otm_levels", [5.0])})
    dte_levels = sorted({int(x) for x in short_put_config.get("dte_levels", [20])})
    rounding_rule = short_put_config.get("strike_rounding", "No rounding")
    suppress_overlap = bool(short_put_config.get("one_active_per_ticker", False))

    dates = pd.to_datetime(work["Date"]).to_numpy(dtype="datetime64[ns]")
    opens = work["Open"].to_numpy(dtype=float)
    highs = work["High"].to_numpy(dtype=float)
    lows = work["Low"].to_numpy(dtype=float)
    closes = work["Close"].to_numpy(dtype=float)

    signal_positions = np.flatnonzero(signal.to_numpy(dtype=bool))
    rows = []

    # Overlap tracking is per OTM/DTE combination so comparisons remain fair.
    active_until: dict[tuple[float, int], int] = {}

    for i in signal_positions:
        entry_i = int(i) + 1
        if entry_i >= len(work):
            continue

        entry_price = opens[entry_i]
        if not np.isfinite(entry_price) or entry_price <= 0:
            continue

        signal_date = pd.Timestamp(dates[i])
        entry_date = pd.Timestamp(dates[entry_i])
        snapshot = _signal_snapshot(work, int(i))

        for dte in dte_levels:
            target_date = entry_date + pd.Timedelta(days=int(dte))

            # Critical: do not use the dataset's final row as a fake expiry
            # if history ends before the requested target date.
            if pd.Timestamp(dates[-1]) < target_date:
                continue

            target64 = np.datetime64(target_date.to_datetime64())
            expiry_i = int(np.searchsorted(dates, target64, side="right") - 1)
            if expiry_i < entry_i:
                continue

            expiry_date = pd.Timestamp(dates[expiry_i])
            period_lows = lows[entry_i:expiry_i + 1]
            period_highs = highs[entry_i:expiry_i + 1]

            if len(period_lows) == 0:
                continue

            lowest_price = float(np.nanmin(period_lows))
            highest_price = float(np.nanmax(period_highs))
            expiry_close = float(closes[expiry_i])

            if not all(np.isfinite(x) for x in [lowest_price, highest_price, expiry_close]):
                continue

            for otm in otm_levels:
                key = (otm, dte)
                if suppress_overlap and entry_i <= active_until.get(key, -1):
                    continue

                raw_strike = entry_price * (1.0 - otm / 100.0)
                strike = round_strike_down(raw_strike, rounding_rule)
                if not np.isfinite(strike) or strike <= 0:
                    continue

                touch_mask = period_lows <= strike
                strike_touched = bool(np.any(touch_mask))

                first_touch_date = pd.NaT
                days_to_first_touch = np.nan
                days_before_expiry_at_touch = np.nan
                if strike_touched:
                    rel_touch = int(np.argmax(touch_mask))
                    touch_i = entry_i + rel_touch
                    first_touch_date = pd.Timestamp(dates[touch_i])
                    days_to_first_touch = int((first_touch_date - entry_date).days)
                    days_before_expiry_at_touch = int((expiry_date - first_touch_date).days)

                if expiry_close < strike - PRICE_EQUAL_TOLERANCE:
                    expiry_state = "ITM / Assignment Proxy"
                    assigned = True
                elif expiry_close > strike + PRICE_EQUAL_TOLERANCE:
                    expiry_state = "OTM"
                    assigned = False
                else:
                    expiry_state = "At Strike"
                    assigned = False

                expiry_margin = (expiry_close / strike - 1.0) * 100.0
                underlying_return = (expiry_close / entry_price - 1.0) * 100.0
                mae = (lowest_price / entry_price - 1.0) * 100.0
                mfe = (highest_price / entry_price - 1.0) * 100.0
                strike_low_pct = (lowest_price / strike - 1.0) * 100.0
                max_breach = min(strike_low_pct, 0.0)

                out = {
                    "Ticker": ticker,
                    "SignalDate": signal_date,
                    "EntryDate": entry_date,
                    "UnderlyingEntryPrice": entry_price,
                    "OTM_Pct": otm,
                    "RawSyntheticStrike": raw_strike,
                    "PutStrike": strike,
                    "StrikeRounding": rounding_rule,
                    "TargetDTE_CalendarDays": dte,
                    "TargetExpiryDate": target_date,
                    "ActualExpiryDate": expiry_date,
                    "ActualCalendarDays": int((expiry_date - entry_date).days),
                    "ExpiryClose": expiry_close,
                    "ExpiryState": expiry_state,
                    "AssignmentProxy": assigned,
                    "StrikeTouched": strike_touched,
                    "FirstStrikeTouchDate": first_touch_date,
                    "CalendarDaysToFirstTouch": days_to_first_touch,
                    "CalendarDaysBeforeExpiryAtFirstTouch": days_before_expiry_at_touch,
                    "LowestPriceDuringTrade": lowest_price,
                    "HighestPriceDuringTrade": highest_price,
                    "ExpiryMarginPct": expiry_margin,
                    "UnderlyingReturnPct": underlying_return,
                    "MAE_Pct": mae,
                    "MFE_Pct": mfe,
                    "LowestVsStrikePct": strike_low_pct,
                    "MaxStrikeBreachPct": max_breach,
                    "RecoveryClassification": _classify_recovery(strike_touched, expiry_state),
                }
                out.update(snapshot)
                rows.append(out)

                if suppress_overlap:
                    active_until[key] = expiry_i

    return pd.DataFrame(rows), work
