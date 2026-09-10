from __future__ import annotations
from pathlib import Path
from io import BytesIO
import json
import tempfile
import zipfile
import pandas as pd
import streamlit as st

from data_loader import discover_tickers, classify_ticker, available_date_range, load_selected, validate_frame
from backtest_engine import signal_study_for_ticker, trade_backtest_for_ticker
from metrics import (
    summarize_signal_study, summarize_trades, by_ticker,
    summarize_short_puts, short_put_by_otm_dte, short_put_by_otm,
    short_put_by_dte, short_put_by_ticker, short_put_recovery_analysis,
)
from charts import candlestick_chart, short_put_chart
from strategy_storage import save_strategy, list_strategies, load_strategy
from strategy_engine import describe_rules
from candlesticks import PATTERNS
from indicators import sma
from short_put_engine import short_puts_for_ticker
from results_storage import create_run_folder

st.set_page_config(page_title="Parquet Backtest Engine", layout="wide")
st.title("📈 Parquet Backtest Engine")
st.caption(
    "GitHub / Streamlit Cloud edition. Daily-bar signals are known only after "
    "the signal-day close; default entry is the next trading day's open."
)
st.info(
    "For a public app, start with a small ticker selection before running the "
    "full 500+ stock universe. Large OTM × DTE comparisons can create millions "
    "of synthetic observations and use substantial memory."
)

# -------------------------------------------------------------------------
# CLOUD-READY PATHS
# -------------------------------------------------------------------------
# This file can run in:
#   1) GitHub + Streamlit Community Cloud
#   2) your own local Linux/Windows/Mac machine
#
# Put your Parquet files in:
#     market_data/prices/
# beside this app.py file.
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "market_data" / "prices"

# Streamlit Community Cloud's local filesystem should be treated as temporary.
# We still create runtime folders because some existing functions write files,
# but the user can download the current run directly from the browser.
RUNTIME_DIR = Path(tempfile.gettempdir()) / "ray_backtest_streamlit"
DEFAULT_RESULTS_DIR = RUNTIME_DIR / "backtest_results"
DEFAULT_STRATEGY_DIR = RUNTIME_DIR / "strategies"
DEFAULT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_STRATEGY_DIR.mkdir(parents=True, exist_ok=True)

with st.sidebar:
    st.header("Data")
    st.caption("Cloud data path is relative to this GitHub repository.")
    data_dir = st.text_input("Parquet folder", str(DEFAULT_DATA_DIR))
    results_dir = st.text_input(
        "Temporary results folder",
        str(DEFAULT_RESULTS_DIR),
        help=(
            "On Streamlit Community Cloud this folder is temporary. "
            "Use the download buttons after each run to keep your results."
        ),
    )
    if st.button("Clear cached price data"):
        st.cache_data.clear()
        st.success("Streamlit data cache cleared.")


def _csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _json_bytes(obj: dict) -> bytes:
    return json.dumps(obj, indent=2, default=str).encode("utf-8")


def _build_result_zip(
    result_df: pd.DataFrame,
    run_mode: str,
    config: dict,
    error_df: pd.DataFrame,
    quality_df: pd.DataFrame,
) -> bytes:
    """Create an in-memory ZIP so Streamlit Cloud users can save a run."""
    buffer = BytesIO()

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        if run_mode == "Signal Study":
            zf.writestr("signals.csv", _csv_bytes(result_df))

            if not result_df.empty:
                overall = summarize_signal_study(result_df)
                zf.writestr("overall_summary.csv", _csv_bytes(overall))

                rows = []
                for ticker_name, grp in result_df.groupby("Ticker"):
                    for _, row in summarize_signal_study(grp).iterrows():
                        item = row.to_dict()
                        item["Ticker"] = ticker_name
                        rows.append(item)
                zf.writestr(
                    "by_ticker.csv",
                    _csv_bytes(pd.DataFrame(rows)),
                )

        elif run_mode == "Trade Backtest":
            zf.writestr("trades.csv", _csv_bytes(result_df))

            if not result_df.empty:
                zf.writestr(
                    "overall_summary.csv",
                    _csv_bytes(pd.DataFrame([summarize_trades(result_df)])),
                )
                zf.writestr(
                    "by_ticker.csv",
                    _csv_bytes(by_ticker(result_df)),
                )

        else:
            zf.writestr("short_put_trades.csv", _csv_bytes(result_df))

            if not result_df.empty:
                zf.writestr(
                    "overall_summary.csv",
                    _csv_bytes(pd.DataFrame([summarize_short_puts(result_df)])),
                )
                zf.writestr(
                    "by_otm_dte.csv",
                    _csv_bytes(short_put_by_otm_dte(result_df)),
                )
                zf.writestr(
                    "by_otm.csv",
                    _csv_bytes(short_put_by_otm(result_df)),
                )
                zf.writestr(
                    "by_dte.csv",
                    _csv_bytes(short_put_by_dte(result_df)),
                )
                zf.writestr(
                    "by_ticker.csv",
                    _csv_bytes(short_put_by_ticker(result_df)),
                )
                zf.writestr(
                    "recovery_analysis.csv",
                    _csv_bytes(short_put_recovery_analysis(result_df)),
                )

        zf.writestr("configuration.json", _json_bytes(config))

        if not error_df.empty:
            zf.writestr("errors.csv", _csv_bytes(error_df))

        if not quality_df.empty:
            zf.writestr("data_quality.csv", _csv_bytes(quality_df))

    buffer.seek(0)
    return buffer.getvalue()


tickers = discover_tickers(data_dir)
if not tickers:
    st.error(
        "No .parquet files were found. For GitHub / Streamlit Cloud, put your "
        "files under: market_data/prices/ in the repository."
    )
    st.code(
        "your-repo/\n"
        "├── app.py\n"
        "├── requirements.txt\n"
        "└── market_data/\n"
        "    └── prices/\n"
        "        ├── AAPL.parquet\n"
        "        ├── SPY.parquet\n"
        "        └── ..."
    )
    st.stop()

st.header("1. Stock Universe")
universe_type = st.radio("Universe type", ["All securities", "Stocks only", "ETFs only"], horizontal=True)
available = tickers
if universe_type == "Stocks only":
    available = [t for t in tickers if classify_ticker(t) == "Stock"]
elif universe_type == "ETFs only":
    available = [t for t in tickers if classify_ticker(t) == "ETF"]

c1, c2 = st.columns([1, 4])
with c1:
    select_all = st.checkbox("Select all", value=False)
with c2:
    defaults = available if select_all else [t for t in ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"] if t in available]
    selected = st.multiselect("Tickers", available, default=defaults)
st.write(f"Selected: **{len(selected)}**")
if not selected:
    st.warning("Select at least one ticker.")
    st.stop()

st.header("2. Date Range")
min_date, max_date = available_date_range(data_dir, selected)
use_all_dates = st.checkbox("Use all available dates", value=True)
if use_all_dates:
    start_date, end_date = min_date, max_date
    st.write(f"Selected-universe range: **{min_date} → {max_date}**")
else:
    dc1, dc2 = st.columns(2)
    start_date = dc1.date_input("Start date", min_date, min_value=min_date, max_value=max_date)
    end_date = dc2.date_input("End date", max_date, min_value=min_date, max_value=max_date)

st.header("3. Technical Indicators")
tech = {}
with st.expander("Stochastic Oscillator", expanded=True):
    tech["stoch_enabled"] = st.checkbox("Use Stochastic", value=True)
    a, b, c, d, e = st.columns(5)
    tech["stoch_lookback"] = a.number_input("Lookback", 2, 100, 14)
    tech["stoch_k"] = b.number_input("%K smooth", 1, 20, 3)
    tech["stoch_d"] = c.number_input("%D smooth", 1, 20, 3)
    tech["stoch_op"] = d.selectbox("Slow %K condition", ["<", "<=", ">", ">="])
    tech["stoch_threshold"] = e.number_input("Threshold", -100.0, 100.0, 40.0)

with st.expander("RSI / Williams %R / CCI / ATR / ROC"):
    col1, col2, col3 = st.columns(3)
    with col1:
        tech["rsi_enabled"] = st.checkbox("Use RSI")
        tech["rsi_period"] = st.number_input("RSI period", 2, 100, 14)
        tech["rsi_op"] = st.selectbox("RSI condition", ["<", "<=", ">", ">="])
        tech["rsi_threshold"] = st.number_input("RSI threshold", -200.0, 200.0, 30.0)
        tech["williams_enabled"] = st.checkbox("Use Williams %R")
        tech["williams_period"] = st.number_input("Williams period", 2, 100, 14)
        tech["williams_op"] = st.selectbox("Williams condition", ["<", "<=", ">", ">="])
        tech["williams_threshold"] = st.number_input("Williams threshold", -100.0, 0.0, -80.0)
    with col2:
        tech["cci_enabled"] = st.checkbox("Use CCI")
        tech["cci_period"] = st.number_input("CCI period", 2, 100, 20)
        tech["cci_op"] = st.selectbox("CCI condition", ["<", "<=", ">", ">="])
        tech["cci_threshold"] = st.number_input("CCI threshold", -500.0, 500.0, -100.0)
        tech["atr_enabled"] = st.checkbox("Use ATR %")
        tech["atr_period"] = st.number_input("ATR period", 2, 100, 14)
        tech["atr_op"] = st.selectbox("ATR% condition", ["<", "<=", ">", ">="])
        tech["atr_threshold"] = st.number_input("ATR % threshold", 0.0, 100.0, 4.0)
    with col3:
        tech["roc_enabled"] = st.checkbox("Use ROC")
        tech["roc_period"] = st.number_input("ROC period", 1, 252, 20)
        tech["roc_op"] = st.selectbox("ROC condition", ["<", "<=", ">", ">="])
        tech["roc_threshold"] = st.number_input("ROC threshold %", -100.0, 500.0, 0.0)

with st.expander("MACD / Bollinger Bands / ADX"):
    col1, col2, col3 = st.columns(3)
    with col1:
        tech["macd_enabled"] = st.checkbox("Use MACD")
        tech["macd_fast"] = st.number_input("MACD fast", 2, 100, 12)
        tech["macd_slow"] = st.number_input("MACD slow", 3, 200, 26)
        tech["macd_signal"] = st.number_input("MACD signal", 1, 50, 9)
        tech["macd_rule"] = st.selectbox("MACD rule", ["MACD > Signal", "MACD < Signal", "Bullish crossover", "Bearish crossover", "MACD > 0", "MACD < 0"])
    with col2:
        tech["boll_enabled"] = st.checkbox("Use Bollinger Bands")
        tech["boll_period"] = st.number_input("Bollinger period", 2, 200, 20)
        tech["boll_std"] = st.number_input("Std dev multiplier", 0.5, 5.0, 2.0, step=0.1)
        tech["boll_rule"] = st.selectbox("Bollinger rule", ["Close below lower band", "Close above upper band"])
    with col3:
        tech["adx_enabled"] = st.checkbox("Use ADX")
        tech["adx_period"] = st.number_input("ADX period", 2, 100, 14)
        tech["adx_op"] = st.selectbox("ADX condition", ["<", "<=", ">", ">="], index=2)
        tech["adx_threshold"] = st.number_input("ADX threshold", 0.0, 100.0, 20.0)

st.header("4. Candlestick Patterns")
cand_patterns = st.multiselect("Patterns", list(PATTERNS.keys()), default=["Bullish Engulfing"])
cand_logic = st.radio("If multiple patterns selected", ["ANY", "ALL"], horizontal=True)

st.header("5. Price Action")
pa = {}
with st.expander("Price action filters"):
    col1, col2, col3 = st.columns(3)
    with col1:
        pa["return_enabled"] = st.checkbox("Previous return")
        pa["return_days"] = st.number_input("Return lookback days", 1, 252, 20)
        pa["return_op"] = st.selectbox("Return condition", ["<", "<=", ">", ">="])
        pa["return_threshold"] = st.number_input("Return threshold %", -100.0, 500.0, 0.0)
        pa["consecutive_enabled"] = st.checkbox("Consecutive closes")
        pa["consecutive_direction"] = st.selectbox("Direction", ["Up", "Down"])
        pa["consecutive_days"] = st.number_input("Consecutive days", 2, 20, 3)
    with col2:
        pa["gap_enabled"] = st.checkbox("Gap filter")
        pa["gap_op"] = st.selectbox("Gap condition", ["<", "<=", ">", ">="])
        pa["gap_threshold"] = st.number_input("Gap %", -50.0, 50.0, 1.0)
        pa["distance_low_enabled"] = st.checkbox("Within % of recent low")
        pa["distance_low_period"] = st.number_input("Low lookback", 2, 300, 20)
        pa["distance_low_pct"] = st.number_input("Within low %", 0.0, 100.0, 2.0)
        pa["distance_high_enabled"] = st.checkbox("Within % of recent high")
        pa["distance_high_period"] = st.number_input("High lookback", 2, 300, 20)
        pa["distance_high_pct"] = st.number_input("Within high %", 0.0, 100.0, 2.0)
    with col3:
        pa["breakout_enabled"] = st.checkbox("Breakout above prior high")
        pa["breakout_period"] = st.number_input("Breakout lookback", 2, 300, 20)
        pa["breakdown_enabled"] = st.checkbox("Breakdown below prior low")
        pa["breakdown_period"] = st.number_input("Breakdown lookback", 2, 300, 20)
        pa["distance_ma_enabled"] = st.checkbox("Within % of MA")
        pa["distance_ma_period"] = st.selectbox("MA proximity period", [20, 50, 100, 200], index=3)
        pa["distance_ma_pct"] = st.number_input("Within MA %", 0.0, 50.0, 2.0)

st.header("6. Volume Criteria")
vol = {"enabled": st.checkbox("Use volume filter")}
if vol["enabled"]:
    vc1, vc2, vc3 = st.columns(3)
    vol["avg_period"] = vc1.selectbox("Average volume period", [5, 10, 20, 50], index=2)
    vol["rule"] = vc2.selectbox("Volume rule", ["Volume > average", "Volume < average", "Relative volume >", "Volume increasing N days"])
    vol["relative_threshold"] = vc3.number_input("Relative volume threshold", 0.0, 10.0, 1.5, step=0.1)
    vol["increasing_days"] = vc3.number_input("Increasing days", 2, 20, 3)

st.header("7. Stock Moving Average & Market Regime")
ma = {"enabled": st.checkbox("Use stock MA filter")}
if ma["enabled"]:
    mc1, mc2, mc3 = st.columns(3)
    ma["rule"] = mc1.selectbox("Stock MA rule", ["Close > MA", "Close < MA", "MA20 > MA50", "MA50 > MA200", "MA20 > MA50 > MA200", "MA200 rising"])
    ma["period"] = mc2.selectbox("MA period", [20, 50, 100, 200], index=3)
    ma["slope_lookback"] = mc3.number_input("MA slope lookback", 1, 100, 20)

market = {"enabled": st.checkbox("Use SPY / QQQ / IWM / GLD market filter")}
if market["enabled"]:
    bc1, bc2, bc3, bc4 = st.columns(4)
    benchmark_options = [x for x in ["SPY", "QQQ", "IWM", "GLD"] if x in tickers]
    if not benchmark_options:
        st.error("No SPY/QQQ/IWM/GLD parquet file found.")
        market["enabled"] = False
    else:
        market["benchmark"] = bc1.selectbox("Benchmark", benchmark_options)
        market["rule"] = bc2.selectbox("Benchmark rule", ["Close > MA", "Close < MA", "MA20 > MA50", "MA50 > MA200", "20-day return >", "20-day return <"])
        market["period"] = bc3.selectbox("Benchmark MA", [20, 50, 100, 200], index=3)
        market["return_threshold"] = bc4.number_input("20D return threshold %", -100.0, 500.0, 0.0)

st.header("8. Rule Logic")
global_logic = st.radio("Combine selected rule groups using", ["ALL", "ANY"], horizontal=True, help="ALL = AND. ANY = OR.")

st.header("9. Backtest Mode / Options")
mode = st.radio(
    "Mode",
    ["Signal Study", "Trade Backtest", "Short Put Analysis"],
    horizontal=True,
)

exit_cfg = {"holding_days": 5, "profit_target_pct": None, "stop_loss_pct": None}
execution = {"one_trade_per_ticker": True}
short_put_cfg = {
    "otm_levels": [5.0],
    "dte_levels": [20],
    "strike_rounding": "No rounding",
    "one_active_per_ticker": False,
}

if mode == "Trade Backtest":
    ec1, ec2, ec3 = st.columns(3)
    exit_cfg["holding_days"] = ec1.number_input("Maximum holding days", 1, 252, 5)
    use_target = ec2.checkbox("Use profit target")
    target_val = ec2.number_input("Profit target %", 0.1, 100.0, 3.0)
    exit_cfg["profit_target_pct"] = target_val if use_target else None
    use_stop = ec3.checkbox("Use stop loss")
    stop_val = ec3.number_input("Stop loss %", 0.1, 100.0, 2.0)
    exit_cfg["stop_loss_pct"] = stop_val if use_stop else None
    execution["one_trade_per_ticker"] = st.checkbox("Only one active trade per ticker", value=True)
    st.info("If target and stop are both touched in the same daily bar, the engine conservatively assumes the stop was hit first.")

elif mode == "Short Put Analysis":
    st.subheader("Options / Short Put Analysis")
    st.warning(
        "This is an underlying-price proxy, not an historical option-chain P&L test. "
        "Expiry Close below the synthetic strike is labelled 'Expiration ITM / Assignment Proxy'."
    )

    pc1, pc2, pc3 = st.columns(3)

    with pc1:
        test_multi_otm = st.checkbox("Test multiple OTM distances", value=False)
        if test_multi_otm:
            short_put_cfg["otm_levels"] = st.multiselect(
                "OTM distances (%)",
                [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0],
                default=[3.0, 5.0, 7.0, 10.0],
            )
        else:
            short_put_cfg["otm_levels"] = [
                float(st.number_input("Put strike OTM %", 0.1, 50.0, 5.0, step=0.5))
            ]

    with pc2:
        test_multi_dte = st.checkbox("Test multiple DTE values", value=False)
        if test_multi_dte:
            short_put_cfg["dte_levels"] = st.multiselect(
                "Calendar DTE",
                [7, 14, 20, 21, 30, 45, 60],
                default=[7, 14, 20, 30, 45],
            )
        else:
            short_put_cfg["dte_levels"] = [
                int(st.number_input("Calendar DTE", 1, 365, 20))
            ]

    with pc3:
        short_put_cfg["strike_rounding"] = st.selectbox(
            "Synthetic strike rounding",
            [
                "No rounding",
                "Nearest $0.50 down",
                "Nearest $1 down",
                "Nearest $2.50 down",
                "Nearest $5 down",
            ],
        )
        short_put_cfg["one_active_per_ticker"] = st.checkbox(
            "Suppress overlapping puts per ticker",
            value=False,
            help=(
                "For research, leaving this OFF treats every qualifying signal independently. "
                "Turn it ON if you want only one active synthetic put per ticker for each OTM/DTE combination."
            ),
        )

    if not short_put_cfg["otm_levels"] or not short_put_cfg["dte_levels"]:
        st.error("Select at least one OTM level and one DTE value.")
        st.stop()

    combinations = len(short_put_cfg["otm_levels"]) * len(short_put_cfg["dte_levels"])
    st.caption(
        f"Each qualifying signal will test {combinations} OTM/DTE combination(s). "
        "DTE means calendar days; weekends/holidays use the last trading day on or before the target date."
    )

config = {
    "technical": tech,
    "candlesticks": {"patterns": cand_patterns, "logic": cand_logic},
    "price_action": pa,
    "volume": vol,
    "moving_average": ma,
    "market": market,
    "global_logic": global_logic,
    "exit": exit_cfg,
    "execution": execution,
    "short_put": short_put_cfg,
    "test_start": str(start_date) if start_date else None,
    "test_end": str(end_date) if end_date else None,
}

with st.expander("Current strategy summary", expanded=True):
    for rule in describe_rules(config):
        st.write("•", rule)
    if mode == "Short Put Analysis":
        st.write(
            "• Short Put:",
            f"OTM {short_put_cfg['otm_levels']}% | "
            f"DTE {short_put_cfg['dte_levels']} calendar days | "
            f"{short_put_cfg['strike_rounding']}"
        )
    st.json(config)

st.header("10. Save Strategy")
st.caption("Saved strategies are temporary on Streamlit Community Cloud. Download the configuration JSON if you want to keep it permanently.")
sc1, sc2 = st.columns(2)
strategy_name = sc1.text_input("Strategy name", "Stoch40_BullishEngulfing")
sc2.download_button(
    "⬇ Download current configuration JSON",
    data=_json_bytes(config),
    file_name=f"{strategy_name}_configuration.json",
    mime="application/json",
)
if sc1.button("Save current strategy"):
    p = save_strategy(DEFAULT_STRATEGY_DIR, strategy_name, config)
    st.success(f"Saved for this running session: {p}")
saved = list_strategies(DEFAULT_STRATEGY_DIR)
if saved:
    chosen = sc2.selectbox("Saved strategies", [""] + [p.name for p in saved])
    if chosen and sc2.button("Show saved JSON"):
        st.json(load_strategy(Path(DEFAULT_STRATEGY_DIR) / chosen))

st.header("11. Run Backtest")
if st.button("▶ Run", type="primary"):
    benchmark_tickers = {market["benchmark"]} if market.get("enabled") else set()
    load_tickers = sorted(set(selected) | benchmark_tickers)

    # Signals are limited by config.test_end, but short-put trades need forward
    # price history through expiry. Load a forward buffer only for this mode.
    data_load_end = end_date
    if mode == "Short Put Analysis":
        max_dte = max(short_put_cfg["dte_levels"])
        data_load_end = pd.Timestamp(end_date) + pd.Timedelta(days=max_dte + 10)

    # Load all pre-start history for indicator warm-up.
    frames, errors = load_selected(data_dir, load_tickers, None, data_load_end)
    benchmark_frames = {t: frames[t] for t in benchmark_tickers if t in frames}
    results, quality = [], []
    progress = st.progress(0)

    for ix, ticker in enumerate(selected):
        if ticker not in frames:
            continue
        quality += validate_frame(ticker, frames[ticker])
        try:
            if mode == "Signal Study":
                r, _ = signal_study_for_ticker(
                    ticker, frames[ticker], config, benchmark_frames
                )
            elif mode == "Trade Backtest":
                r, _ = trade_backtest_for_ticker(
                    ticker, frames[ticker], config, benchmark_frames
                )
            else:
                r, _ = short_puts_for_ticker(
                    ticker, frames[ticker], config, benchmark_frames, short_put_cfg
                )
            if not r.empty:
                results.append(r)
        except Exception as exc:
            errors.append({"Ticker": ticker, "Error": str(exc)})
        progress.progress((ix + 1) / len(selected))

    result_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    error_df = pd.DataFrame(errors)
    quality_df = pd.DataFrame(quality)

    # Every run gets its own folder, preventing accidental overwrite.
    run_dir = create_run_folder(results_dir, strategy_name, mode)
    if not result_df.empty:
        if mode == "Signal Study":
            result_df.to_csv(run_dir / "signals.csv", index=False)
            summarize_signal_study(result_df).to_csv(
                run_dir / "overall_summary.csv", index=False
            )
            ticker_rows = []
            for ticker_name, grp in result_df.groupby("Ticker"):
                for _, r in summarize_signal_study(grp).iterrows():
                    rowx = r.to_dict()
                    rowx["Ticker"] = ticker_name
                    ticker_rows.append(rowx)
            pd.DataFrame(ticker_rows).to_csv(
                run_dir / "by_ticker.csv", index=False
            )

        elif mode == "Trade Backtest":
            result_df.to_csv(run_dir / "trades.csv", index=False)
            pd.DataFrame([summarize_trades(result_df)]).to_csv(
                run_dir / "overall_summary.csv", index=False
            )
            by_ticker(result_df).to_csv(run_dir / "by_ticker.csv", index=False)

        else:
            result_df.to_csv(run_dir / "short_put_trades.csv", index=False)
            pd.DataFrame([summarize_short_puts(result_df)]).to_csv(
                run_dir / "overall_summary.csv", index=False
            )
            short_put_by_otm_dte(result_df).to_csv(
                run_dir / "by_otm_dte.csv", index=False
            )
            short_put_by_otm(result_df).to_csv(
                run_dir / "by_otm.csv", index=False
            )
            short_put_by_dte(result_df).to_csv(
                run_dir / "by_dte.csv", index=False
            )
            short_put_by_ticker(result_df).to_csv(
                run_dir / "by_ticker.csv", index=False
            )
            short_put_recovery_analysis(result_df).to_csv(
                run_dir / "recovery_analysis.csv", index=False
            )

    (run_dir / "configuration.json").write_text(
        json.dumps(config, indent=2, default=str), encoding="utf-8"
    )
    if not error_df.empty:
        error_df.to_csv(run_dir / "errors.csv", index=False)
    if not quality_df.empty:
        quality_df.to_csv(run_dir / "data_quality.csv", index=False)

    st.session_state["result_df"] = result_df
    st.session_state["run_mode"] = mode
    st.session_state["config"] = config
    st.session_state["errors"] = error_df
    st.session_state["quality"] = quality_df
    st.session_state["run_dir"] = str(run_dir)
    st.session_state["run_end_date"] = str(data_load_end)

if "result_df" in st.session_state:
    result_df = st.session_state["result_df"]
    run_mode = st.session_state["run_mode"]
    run_dir = Path(st.session_state["run_dir"])

    st.header("12. Results")
    st.info(
        "A temporary server copy was created for this session. "
        "Use the download buttons below to keep the results permanently."
    )

    result_zip = _build_result_zip(
        result_df,
        run_mode,
        st.session_state["config"],
        st.session_state["errors"],
        st.session_state["quality"],
    )

    dl1, dl2 = st.columns(2)
    dl1.download_button(
        "⬇ Download complete result package (ZIP)",
        data=result_zip,
        file_name=f"{run_mode.replace(' ', '_').lower()}_results.zip",
        mime="application/zip",
        key="download_full_result_zip",
    )
    dl2.download_button(
        "⬇ Download main result CSV",
        data=_csv_bytes(result_df),
        file_name=(
            "short_put_trades.csv"
            if run_mode == "Short Put Analysis"
            else "trades.csv"
            if run_mode == "Trade Backtest"
            else "signals.csv"
        ),
        mime="text/csv",
        key="download_main_result_csv",
    )

    if result_df.empty:
        st.warning("No qualifying signals/trades found for the selected rules.")

    elif run_mode == "Signal Study":
        st.subheader("Overall signal study")
        st.dataframe(
            summarize_signal_study(result_df),
            use_container_width=True
        )
        st.subheader("Signals")
        st.dataframe(result_df, use_container_width=True, height=450)

    elif run_mode == "Trade Backtest":
        summary = summarize_trades(result_df)
        cols = st.columns(4)
        for i, (k, v) in enumerate(summary.items()):
            cols[i % 4].metric(
                k, f"{v:.2f}" if isinstance(v, float) else str(v)
            )
        st.subheader("By ticker")
        st.dataframe(by_ticker(result_df), use_container_width=True)
        st.subheader("Trade log")
        st.dataframe(result_df, use_container_width=True, height=450)

    else:
        summary = summarize_short_puts(result_df)

        st.subheader("Short Put Risk Summary")
        mc = st.columns(4)
        important = [
            ("Short puts tested", summary.get("ShortPutsTested", 0), ""),
            ("Assignment proxy", summary.get("AssignmentProxy_Pct", 0), "%"),
            ("Expired OTM", summary.get("ExpiredOTM_Pct", 0), "%"),
            ("Strike touched", summary.get("StrikeTouched_Pct", 0), "%"),
            ("Touched & recovered", summary.get("TouchedRecovered_Pct", 0), "%"),
            ("Avg expiry margin", summary.get("AvgExpiryMarginPct", 0), "%"),
            ("Avg MAE", summary.get("AvgMAE_Pct", 0), "%"),
            ("Avg max strike breach", summary.get("AvgMaxStrikeBreachPct", 0), "%"),
        ]
        for i, (label, value, suffix) in enumerate(important):
            if isinstance(value, (int, float)):
                rendered = f"{value:,.2f}{suffix}" if suffix else f"{value:,.0f}"
            else:
                rendered = str(value)
            mc[i % 4].metric(label, rendered)

        st.info(
            "Assignment Proxy = the underlying finished below the synthetic put strike at expiry. "
            "This is not actual historical broker assignment and does not include option premium or IV."
        )

        st.subheader("OTM × DTE comparison")
        compare_df = short_put_by_otm_dte(result_df)
        st.dataframe(compare_df, use_container_width=True, height=400)

        tab1, tab2, tab3 = st.tabs(["By OTM", "By DTE", "Recovery"])
        with tab1:
            st.dataframe(short_put_by_otm(result_df), use_container_width=True)
        with tab2:
            st.dataframe(short_put_by_dte(result_df), use_container_width=True)
        with tab3:
            st.dataframe(
                short_put_recovery_analysis(result_df),
                use_container_width=True
            )

        st.subheader("By ticker")
        minimum_sample = st.number_input(
            "Minimum short puts per ticker",
            min_value=1,
            max_value=10000,
            value=20,
        )
        ticker_table = short_put_by_ticker(result_df)
        if not ticker_table.empty:
            ticker_table = ticker_table[ticker_table["Trades"] >= minimum_sample]
        st.dataframe(ticker_table, use_container_width=True, height=450)

        st.subheader("Synthetic short-put trade log")
        st.dataframe(result_df, use_container_width=True, height=500)

    if not st.session_state["errors"].empty:
        with st.expander("Errors / skipped tickers"):
            st.dataframe(st.session_state["errors"], use_container_width=True)

    if not st.session_state["quality"].empty:
        with st.expander("Data quality notes"):
            st.dataframe(st.session_state["quality"], use_container_width=True)

    if not result_df.empty:
        st.header("13. Inspect a Signal / Trade")
        options = list(result_df.index[:5000])
        chosen_idx = st.selectbox(
            "Result row",
            options,
            format_func=lambda i: (
                f"{i}: {result_df.loc[i,'Ticker']} - "
                f"{pd.Timestamp(result_df.loc[i,'SignalDate']).date()}"
            ),
        )
        row = result_df.loc[chosen_idx]
        ticker = row["Ticker"]

        signal_date = pd.Timestamp(row["SignalDate"])
        if run_mode == "Short Put Analysis":
            chart_end = pd.Timestamp(row["ActualExpiryDate"])
        elif run_mode == "Trade Backtest":
            chart_end = pd.Timestamp(row["ExitDate"])
        else:
            chart_end = signal_date + pd.Timedelta(days=100)

        frames_chart, _ = load_selected(
            data_dir, [ticker], None, chart_end + pd.Timedelta(days=5)
        )
        if ticker in frames_chart:
            chart_df = frames_chart[ticker].copy()
            for n in [20, 50, 200]:
                chart_df[f"MA{n}"] = sma(chart_df["Close"], n)

            display_end = chart_end + pd.Timedelta(days=15)
            chart_df = chart_df[
                (chart_df["Date"] >= signal_date - pd.Timedelta(days=100))
                & (chart_df["Date"] <= display_end)
            ]

            if run_mode == "Short Put Analysis":
                st.plotly_chart(
                    short_put_chart(
                        chart_df,
                        title=(
                            f"{ticker} | {row['OTM_Pct']:.1f}% OTM | "
                            f"{int(row['TargetDTE_CalendarDays'])} DTE | "
                            f"{row['ExpiryState']}"
                        ),
                        signal_date=signal_date,
                        entry_date=row["EntryDate"],
                        expiry_date=row["ActualExpiryDate"],
                        strike=row["PutStrike"],
                        first_touch_date=row["FirstStrikeTouchDate"],
                    ),
                    use_container_width=True,
                )
            else:
                st.plotly_chart(
                    candlestick_chart(
                        chart_df,
                        ticker,
                        signal_date,
                        row.get("EntryDate"),
                        row.get("ExitDate"),
                    ),
                    use_container_width=True,
                )
