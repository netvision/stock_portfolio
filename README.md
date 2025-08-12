# Indian Stock Portfolio Dashboard

A Streamlit dashboard to track an Indian stock portfolio (NSE), analyze performance, generate multi-factor Buy/Hold/Exit signals, and research new symbols for short/long-term suitability.

## Features
- Load portfolio from CSV (symbol, quantity, buy_price)
- Live prices via Yahoo Finance
- KPIs, holdings table, downloadable signals
- Technicals: SMA20/50/200, RSI(14), MACD(12,26,9), ATR(14)
- Fundamentals: PE and Dividend Yield (best-effort)
- Research Tool: analyze any NSE symbol (e.g., RELIANCE or RELIANCE.NS)
- Colorful dark theme and safe charts

## Quick start
1. Create and activate a virtual environment
2. Install requirements
3. Run the app

## CSV format
```
symbol,quantity,buy_price
RELIANCE,10,2500
TCS,5,3500
HDFCBANK,8,1600
```
Suffix .NS is added automatically.

## Notes
Data is best-effort and may differ from broker data. Signals are educational only.

## Deploy for free

### Option 1: Streamlit Community Cloud (free)
1. Push this repo to GitHub (done).
2. Go to https://share.streamlit.io/ and sign in with GitHub.
3. Create new app: pick this repo and branch `main`, main file `portfolio_dashboard.py`.
4. Advanced settings: leave default. The app will pick `requirements.txt` and `runtime.txt` (Python 3.11).
5. Click Deploy. After first build, your app URL will be available to share.

### Option 2: Hugging Face Spaces (free)
1. Create a new Space: type = Streamlit.
2. Connect it to your GitHub repo or upload files.
3. In Space settings, ensure `App file` = `portfolio_dashboard.py`.
4. The Space builds automatically and gives you a public URL.
