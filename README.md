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
