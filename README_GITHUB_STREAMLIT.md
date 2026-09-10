# Ray Backtest Engine — Streamlit Cloud Edition

This package is designed to run both locally and on Streamlit Community Cloud.

## Repository structure

Your GitHub repository should look like:

```text
stock-backtest-engine/
├── app.py
├── backtest_engine.py
├── candlesticks.py
├── charts.py
├── data_loader.py
├── indicators.py
├── market_filters.py
├── metrics.py
├── price_action.py
├── results_storage.py
├── short_put_engine.py
├── strategy_engine.py
├── strategy_storage.py
├── volume_filters.py
├── requirements.txt
├── .gitignore
├── .streamlit/
│   └── config.toml
└── market_data/
    └── prices/
        ├── AAPL.parquet
        ├── MSFT.parquet
        ├── SPY.parquet
        └── ...
```

## Important change from the local-only version

There are no hard-coded paths such as:

```text
/home/siew/backtest/market_data/prices
```

The app automatically uses:

```text
market_data/prices/
```

relative to `app.py`.

This works on both GitHub/Streamlit Cloud and your local machine.

## Step 1 — Test locally first

Copy a few Parquet files into:

```text
market_data/prices/
```

For example:

```text
AAPL.parquet
MSFT.parquet
NVDA.parquet
SPY.parquet
QQQ.parquet
GLD.parquet
```

Activate your environment and run:

```bash
cd /path/to/stock-backtest-engine
pip install -r requirements.txt
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

## Step 2 — Put the project into GitHub

From the project folder:

```bash
git init
git add .
git commit -m "Initial Streamlit cloud backtest engine"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/stock-backtest-engine.git
git push -u origin main
```

If your repository already exists, use its GitHub URL instead.

## Step 3 — Deploy to Streamlit Community Cloud

1. Sign in to Streamlit Community Cloud using GitHub.
2. Create a new app.
3. Select your repository.
4. Branch: `main`
5. Main file path: `app.py`
6. Deploy.

## Step 4 — Add the full Parquet universe

After the small test is working, copy all of your Parquet files into:

```text
market_data/prices/
```

Then:

```bash
git add market_data/prices
git commit -m "Add full market price database"
git push
```

Streamlit Community Cloud will redeploy from the updated repository.

## Cloud result storage

The app writes a temporary result copy while it is running, but Streamlit Cloud
filesystem storage should not be treated as permanent.

After every backtest, use:

- **Download complete result package (ZIP)**, or
- **Download main result CSV**

The ZIP contains the relevant summary tables, trade/signal log,
`configuration.json`, and any data-quality/error files.

## Caching

Each Parquet ticker file is cached by Streamlit after first load. This reduces
repeated disk reads during the same deployment/session.

Use **Clear cached price data** in the sidebar if you replace/update Parquet
files and want the app to reload them.

## Performance recommendation

Start with:
- 1–10 tickers
- one OTM level
- one DTE value

Then increase gradually.

A full 500+ ticker scan combined with many OTM and DTE combinations can create
millions of synthetic short-put rows and consume substantial memory.

## Data format

Each Parquet file should contain at least:

- Date
- Open
- High
- Low
- Close

Optional but useful:
- Adj Close
- Volume
- Dividends
- Stock Splits

File naming should be:

```text
AAPL.parquet
MSFT.parquet
SPY.parquet
```

## Security

Do not put passwords, API keys, brokerage credentials, or private secrets
directly in GitHub source code.

This backtest application does not require a broker connection.
