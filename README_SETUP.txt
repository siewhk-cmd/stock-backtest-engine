LOCAL PARQUET BACKTEST GUI - SETUP
==================================

EXPECTED MARKET DATA
/home/siew/backtest/market_data/prices/*.parquet

RECOMMENDED GUI LOCATION
/home/siew/backtest/gui/

1. COPY THE GUI FILES

Unzip backtest_gui_engine.zip and copy the contents of the backtest_gui_engine folder to:

    /home/siew/backtest/gui/

The result should look like:

    /home/siew/backtest/
        market_data/
            prices/
                AAPL.parquet
                MSFT.parquet
                SPY.parquet
                ...
        gui/
            app.py
            backtest_engine.py
            indicators.py
            candlesticks.py
            data_loader.py
            price_action.py
            volume_filters.py
            market_filters.py
            strategy_engine.py
            metrics.py
            charts.py
            strategy_storage.py
            requirements.txt

2. ACTIVATE YOUR PYTHON ENVIRONMENT

If you already have /home/siew/backtest/myenv:

    cd /home/siew/backtest
    source myenv/bin/activate

If not:

    cd /home/siew/backtest
    python3 -m venv myenv
    source myenv/bin/activate

3. INSTALL REQUIREMENTS

    cd /home/siew/backtest/gui
    pip install -r requirements.txt

4. START THE GUI

    streamlit run app.py

Normally your browser opens automatically. If not, use the local address shown in the terminal, normally:

    http://localhost:8501

5. FIRST TEST

For validation, use:

    Tickers: select 5-10 stocks first
    Date: Use all available dates
    Stochastic: ON
        Lookback = 14
        %K smoothing = 3
        %D smoothing = 3
        Slow %K < 40
    Candlestick: Bullish Engulfing
    Other filters: OFF
    Rule Logic: ALL
    Mode: Signal Study

Press Run.

The engine uses the NEXT TRADING DAY OPEN as the entry and calculates:

    1-day forward return
    3-day forward return
    5-day forward return
    10-day forward return
    20-day forward return
    MFE
    MAE

6. FULL UNIVERSE

After confirming the small test works, enable Select all.

7. OUTPUTS

Results are saved by default in:

    /home/siew/backtest/backtest_results/

Signal Study creates:

    signals.csv
    configuration.json
    errors.csv (only if there are errors)

Trade Backtest creates:

    trades.csv
    configuration.json
    errors.csv (only if there are errors)

Saved strategy JSON files are stored in:

    /home/siew/backtest/strategies/

8. STOP STREAMLIT

In the terminal press:

    Ctrl+C

9. START IT AGAIN LATER

    cd /home/siew/backtest
    source myenv/bin/activate
    cd gui
    streamlit run app.py

IMPORTANT BACKTEST NOTES
------------------------

- Signal-day Close/High/Low/Volume are only treated as known after that day's close.
- Entry is the next trading day's Open.
- Engulfing patterns compare real candle bodies.
- Stock-split boundaries are excluded from engulfing patterns when Stock Splits is available.
- If a daily bar touches both a stop and a profit target, the engine assumes the stop was hit first.
- Testing today's stock universe historically introduces survivorship bias if interpreted as a historical S&P 500 membership test.
- This is a research/backtest tool, not an order-execution system.
