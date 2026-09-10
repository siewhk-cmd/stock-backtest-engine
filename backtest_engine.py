from __future__ import annotations
import numpy as np
import pandas as pd

from indicators import stochastic, rsi, williams_r, cci, macd, bollinger, adx, atr, roc, sma
from candlesticks import apply_patterns
from price_action import previous_return, consecutive_up, consecutive_down, gap_pct, distance_from_low, distance_from_high, breakout, breakdown
from volume_filters import avg_volume, relative_volume, volume_increasing
from market_filters import merge_benchmarks
from strategy_engine import compare, crossed_above, crossed_below, combine_conditions


def prepare_indicators(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    out = df.copy()
    tech, ma, vol, pa = config.get("technical", {}), config.get("moving_average", {}), config.get("volume", {}), config.get("price_action", {})

    if tech.get("stoch_enabled"):
        out["STOCH_RAWK"], out["STOCH_K"], out["STOCH_D"] = stochastic(out, tech["stoch_lookback"], tech["stoch_k"], tech["stoch_d"])
    if tech.get("rsi_enabled"):
        out["RSI"] = rsi(out["Close"], tech["rsi_period"])
    if tech.get("williams_enabled"):
        out["WILLR"] = williams_r(out, tech["williams_period"])
    if tech.get("cci_enabled"):
        out["CCI"] = cci(out, tech["cci_period"])
    if tech.get("macd_enabled"):
        out["MACD"], out["MACD_SIGNAL"], out["MACD_HIST"] = macd(out["Close"], tech["macd_fast"], tech["macd_slow"], tech["macd_signal"])
    if tech.get("boll_enabled"):
        out["BB_MID"], out["BB_UPPER"], out["BB_LOWER"], out["BB_WIDTH"] = bollinger(out["Close"], tech["boll_period"], tech["boll_std"])
    if tech.get("adx_enabled"):
        out["ADX"], out["PLUS_DI"], out["MINUS_DI"] = adx(out, tech["adx_period"])
    if tech.get("atr_enabled"):
        out["ATR"] = atr(out, tech["atr_period"])
        out["ATR_PCT"] = out["ATR"] / out["Close"] * 100
    if tech.get("roc_enabled"):
        out["ROC"] = roc(out["Close"], tech["roc_period"])

    needed_ma = set([20,50,100,200]) if ma.get("enabled") else set()
    if pa.get("distance_ma_enabled"):
        needed_ma.add(pa.get("distance_ma_period", 200))
    for n in needed_ma:
        out[f"MA{n}"] = sma(out["Close"], n)

    if vol.get("enabled"):
        n = vol.get("avg_period", 20)
        out["AVG_VOLUME"] = avg_volume(out, n)
        out["REL_VOLUME"] = relative_volume(out, n)
    return out


def build_signal(df: pd.DataFrame, config: dict, benchmark_frames=None):
    out = prepare_indicators(df, config)
    if benchmark_frames:
        out = merge_benchmarks(out, benchmark_frames)
    conditions = []
    tech, cand, pa = config.get("technical", {}), config.get("candlesticks", {}), config.get("price_action", {})
    vol, ma, market = config.get("volume", {}), config.get("moving_average", {}), config.get("market", {})

    if tech.get("stoch_enabled"): conditions.append(compare(out["STOCH_K"], tech["stoch_op"], tech["stoch_threshold"]))
    if tech.get("rsi_enabled"): conditions.append(compare(out["RSI"], tech["rsi_op"], tech["rsi_threshold"]))
    if tech.get("williams_enabled"): conditions.append(compare(out["WILLR"], tech["williams_op"], tech["williams_threshold"]))
    if tech.get("cci_enabled"): conditions.append(compare(out["CCI"], tech["cci_op"], tech["cci_threshold"]))
    if tech.get("atr_enabled"): conditions.append(compare(out["ATR_PCT"], tech["atr_op"], tech["atr_threshold"]))
    if tech.get("roc_enabled"): conditions.append(compare(out["ROC"], tech["roc_op"], tech["roc_threshold"]))

    if tech.get("macd_enabled"):
        rule = tech.get("macd_rule")
        if rule == "MACD > Signal": conditions.append((out["MACD"] > out["MACD_SIGNAL"]).fillna(False))
        elif rule == "MACD < Signal": conditions.append((out["MACD"] < out["MACD_SIGNAL"]).fillna(False))
        elif rule == "Bullish crossover": conditions.append(crossed_above(out["MACD"], out["MACD_SIGNAL"]))
        elif rule == "Bearish crossover": conditions.append(crossed_below(out["MACD"], out["MACD_SIGNAL"]))
        elif rule == "MACD > 0": conditions.append((out["MACD"] > 0).fillna(False))
        elif rule == "MACD < 0": conditions.append((out["MACD"] < 0).fillna(False))

    if tech.get("boll_enabled"):
        rule = tech.get("boll_rule")
        conditions.append((out["Close"] < out["BB_LOWER"]).fillna(False) if rule == "Close below lower band" else (out["Close"] > out["BB_UPPER"]).fillna(False))
    if tech.get("adx_enabled"): conditions.append(compare(out["ADX"], tech["adx_op"], tech["adx_threshold"]))

    patterns = cand.get("patterns", [])
    if patterns:
        out = apply_patterns(out, patterns)
        pconds = [out[f"PATTERN::{p}"] for p in patterns]
        conditions.append(combine_conditions(pconds, cand.get("logic", "ANY"), out.index))

    if pa.get("return_enabled"):
        out["PA_RETURN"] = previous_return(out, pa["return_days"])
        conditions.append(compare(out["PA_RETURN"], pa["return_op"], pa["return_threshold"]))
    if pa.get("consecutive_enabled"):
        conditions.append(consecutive_up(out, pa["consecutive_days"]) if pa["consecutive_direction"] == "Up" else consecutive_down(out, pa["consecutive_days"]))
    if pa.get("gap_enabled"):
        out["GAP_PCT"] = gap_pct(out)
        conditions.append(compare(out["GAP_PCT"], pa["gap_op"], pa["gap_threshold"]))
    if pa.get("distance_low_enabled"):
        out["DIST_LOW_PCT"] = distance_from_low(out, pa["distance_low_period"])
        conditions.append(out["DIST_LOW_PCT"].le(pa["distance_low_pct"]).fillna(False))
    if pa.get("distance_high_enabled"):
        out["DIST_HIGH_PCT"] = distance_from_high(out, pa["distance_high_period"])
        conditions.append(out["DIST_HIGH_PCT"].abs().le(pa["distance_high_pct"]).fillna(False))
    if pa.get("breakout_enabled"): conditions.append(breakout(out, pa["breakout_period"]))
    if pa.get("breakdown_enabled"): conditions.append(breakdown(out, pa["breakdown_period"]))
    if pa.get("distance_ma_enabled"):
        n = pa["distance_ma_period"]
        out["DIST_MA_PCT"] = (out["Close"] / out[f"MA{n}"] - 1).abs() * 100
        conditions.append(out["DIST_MA_PCT"].le(pa["distance_ma_pct"]).fillna(False))

    if vol.get("enabled"):
        rule = vol.get("rule")
        if rule == "Volume > average": conditions.append((out["Volume"] > out["AVG_VOLUME"]).fillna(False))
        elif rule == "Volume < average": conditions.append((out["Volume"] < out["AVG_VOLUME"]).fillna(False))
        elif rule == "Relative volume >": conditions.append((out["REL_VOLUME"] > vol["relative_threshold"]).fillna(False))
        elif rule == "Volume increasing N days": conditions.append(volume_increasing(out, vol["increasing_days"]))

    if ma.get("enabled"):
        rule = ma.get("rule")
        if rule == "Close > MA": conditions.append((out["Close"] > out[f"MA{ma['period']}"]).fillna(False))
        elif rule == "Close < MA": conditions.append((out["Close"] < out[f"MA{ma['period']}"]).fillna(False))
        elif rule == "MA20 > MA50": conditions.append((out["MA20"] > out["MA50"]).fillna(False))
        elif rule == "MA50 > MA200": conditions.append((out["MA50"] > out["MA200"]).fillna(False))
        elif rule == "MA20 > MA50 > MA200": conditions.append(((out["MA20"] > out["MA50"]) & (out["MA50"] > out["MA200"])).fillna(False))
        elif rule == "MA200 rising": conditions.append((out["MA200"] > out["MA200"].shift(ma.get("slope_lookback",20))).fillna(False))

    if market.get("enabled"):
        t, rule = market["benchmark"], market["rule"]
        close = out[f"{t}_Close"]
        if rule == "Close > MA": conditions.append((close > out[f"{t}_MA{market['period']}"]).fillna(False))
        elif rule == "Close < MA": conditions.append((close < out[f"{t}_MA{market['period']}"]).fillna(False))
        elif rule == "MA20 > MA50": conditions.append((out[f"{t}_MA20"] > out[f"{t}_MA50"]).fillna(False))
        elif rule == "MA50 > MA200": conditions.append((out[f"{t}_MA50"] > out[f"{t}_MA200"]).fillna(False))
        elif rule == "20-day return >": conditions.append((out[f"{t}_RET20"] > market["return_threshold"]).fillna(False))
        elif rule == "20-day return <": conditions.append((out[f"{t}_RET20"] < market["return_threshold"]).fillna(False))

    signal = combine_conditions(conditions, config.get("global_logic", "ALL"), out.index)
    test_start = config.get("test_start")
    test_end = config.get("test_end")
    if test_start:
        signal &= out["Date"].ge(pd.Timestamp(test_start))
    if test_end:
        signal &= out["Date"].le(pd.Timestamp(test_end))
    return out, signal.fillna(False)


def signal_study_for_ticker(ticker, df, config, benchmark_frames, windows=(1,3,5,10,20)):
    work, signal = build_signal(df, config, benchmark_frames)
    rows = []
    for i in work.index[signal]:
        entry_i = i + 1
        if entry_i >= len(work): continue
        entry = work.at[entry_i, "Open"]
        if pd.isna(entry) or entry <= 0: continue
        row = {"Ticker": ticker, "SignalDate": work.at[i,"Date"], "EntryDate": work.at[entry_i,"Date"], "SignalClose": work.at[i,"Close"], "EntryOpen": entry}
        for col in ["STOCH_K","STOCH_D","RSI","WILLR","CCI","MACD","ADX","ATR_PCT","ROC","REL_VOLUME","MA20","MA50","MA200"]:
            if col in work.columns: row[col] = work.at[i,col]
        for w in windows:
            end_i = entry_i + w - 1
            if end_i >= len(work):
                row[f"{w}D_ReturnPct"] = row[f"{w}D_MFE_Pct"] = row[f"{w}D_MAE_Pct"] = np.nan
                continue
            period = work.loc[entry_i:end_i]
            row[f"{w}D_ReturnPct"] = (work.at[end_i,"Close"] / entry - 1) * 100
            row[f"{w}D_MFE_Pct"] = (period["High"].max() / entry - 1) * 100
            row[f"{w}D_MAE_Pct"] = (period["Low"].min() / entry - 1) * 100
        rows.append(row)
    return pd.DataFrame(rows), work


def trade_backtest_for_ticker(ticker, df, config, benchmark_frames):
    work, signal = build_signal(df, config, benchmark_frames)
    hold = int(config.get("exit", {}).get("holding_days", 5))
    target = config.get("exit", {}).get("profit_target_pct")
    stop = config.get("exit", {}).get("stop_loss_pct")
    rows, active_until = [], -1
    for i in work.index[signal]:
        entry_i = i + 1
        if entry_i >= len(work): continue
        if config.get("execution", {}).get("one_trade_per_ticker", True) and entry_i <= active_until: continue
        entry = work.at[entry_i,"Open"]
        if pd.isna(entry) or entry <= 0: continue
        final_i = min(entry_i + hold - 1, len(work)-1)
        exit_i, exit_price, reason = final_i, work.at[final_i,"Close"], f"{hold}D Hold"
        for j in range(entry_i, final_i+1):
            lo, hi = work.at[j,"Low"], work.at[j,"High"]
            stop_hit = stop is not None and lo <= entry*(1-stop/100)
            target_hit = target is not None and hi >= entry*(1+target/100)
            if stop_hit and target_hit:
                exit_i, exit_price, reason = j, entry*(1-stop/100), "Stop first (same-day ambiguity)"; break
            if stop_hit:
                exit_i, exit_price, reason = j, entry*(1-stop/100), "Stop Loss"; break
            if target_hit:
                exit_i, exit_price, reason = j, entry*(1+target/100), "Profit Target"; break
        period = work.loc[entry_i:exit_i]
        rows.append({"Ticker":ticker,"SignalDate":work.at[i,"Date"],"EntryDate":work.at[entry_i,"Date"],"EntryPrice":entry,
                     "ExitDate":work.at[exit_i,"Date"],"ExitPrice":exit_price,"ExitReason":reason,"HoldingBars":int(exit_i-entry_i+1),
                     "ReturnPct":(exit_price/entry-1)*100,"MFE_Pct":(period["High"].max()/entry-1)*100,"MAE_Pct":(period["Low"].min()/entry-1)*100})
        active_until = exit_i
    return pd.DataFrame(rows), work
