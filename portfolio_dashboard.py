import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import os
from datetime import datetime
from mstock_client import MStockClient, MStockConfig

# Page setup
st.set_page_config(page_title='Indian Stock Portfolio Dashboard', layout='wide')
st.title('📊 Indian Stock Portfolio Dashboard')

# Colorful styling
st.markdown(
    """
    <style>
    .pos {color:#21c55d; font-weight:600}
    .neg {color:#ef4444; font-weight:600}
    .badge {display:inline-block; padding:2px 8px; border-radius:12px; font-size:0.85rem; margin-right:6px}
    .badge-buy {background:#064e3b; color:#22c55e}
    .badge-hold {background:#1f2937; color:#93c5fd}
    .badge-exit {background:#3f1d1d; color:#f87171}
    .kpi .stMetric {background:#1b1f2a; padding:10px; border-radius:8px}
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar: controls
with st.sidebar:
    st.header('Data Source')
    uploaded_file = st.file_uploader('Upload portfolio.csv', type=['csv'])
    use_local = st.checkbox('Use local portfolio.csv (app folder)', value=True)
    analysis_period = st.selectbox('History period for analysis', ['6mo', '1y', '2y', '5y'], index=1)
    refresh = st.button('Refresh live data')
    st.markdown('---')
    st.header('Signal Weights')
    w_trend = st.slider('Trend (SMA) weight', 0, 4, 2)
    w_momentum = st.slider('Momentum (RSI/MACD) weight', 0, 4, 2)
    w_breakout = st.slider('Breakout proximity weight', 0, 3, 1)
    w_value = st.slider('Valuation (PE/DY) weight', 0, 3, 1)
    w_risk = st.slider('Risk (drawdown/stop) weight', 0, 3, 1)
    st.markdown('---')
    st.header('Research Tool')
    research_symbol_input = st.text_input('Enter symbol to research (e.g., RELIANCE or RELIANCE.NS)')
    research_horizon = st.selectbox('Horizon', ['Short-term', 'Long-term'])
    research_go = st.button('Analyze stock')
    st.markdown('---')
    st.header('Broker (mStock)')
    # Load defaults from secrets if available
    def _sec(path, default=""):
        try:
            ref = st.secrets
            for p in path.split('.'):
                ref = ref[p]
            return ref
        except Exception:
            return default
    m_base_url = st.text_input('Base URL', value=_sec('mstock.base_url', 'https://api.example.com'))
    m_token = st.text_input('Access Token', value=_sec('mstock.access_token', ''), type='password')
    m_client_id = st.text_input('Client ID (optional)', value=_sec('mstock.client_id', ''))
    m_dry_run = st.checkbox('Dry run (paper trade)', value=True)
    m_connect = st.button('Test connection')

# Load portfolio
portfolio_df = None
if uploaded_file is not None:
    portfolio_df = pd.read_csv(uploaded_file)
elif use_local:
    try:
        portfolio_df = pd.read_csv('portfolio.csv')
        st.info('Loaded local portfolio.csv')
    except Exception:
        portfolio_df = None

if portfolio_df is None:
    st.warning('Provide a portfolio CSV with columns: symbol, quantity, buy_price')
    st.stop()

# Clean columns and types
portfolio_df.columns = [c.strip().lower() for c in portfolio_df.columns]
required_cols = {'symbol', 'quantity', 'buy_price'}
missing = required_cols - set(portfolio_df.columns)
if missing:
    st.error(f'Missing required columns: {", ".join(sorted(missing))}')
    st.stop()
portfolio_df['quantity'] = pd.to_numeric(portfolio_df['quantity'], errors='coerce').fillna(0).astype(int)
portfolio_df['buy_price'] = pd.to_numeric(portfolio_df['buy_price'], errors='coerce')

@st.cache_data(ttl=300, show_spinner=False)
def fetch_live_price(symbol: str) -> float:
    sym = symbol if symbol.endswith('.NS') else symbol + '.NS'
    try:
        t = yf.Ticker(sym)
        # Try fast_info when available
        fi = getattr(t, 'fast_info', None)
        if fi is not None:
            last_price = None
            for key in ['last_price', 'lastPrice']:
                val = getattr(fi, key, None) if hasattr(fi, key) else (fi.get(key) if hasattr(fi, 'get') else None)
                if val:
                    last_price = float(val)
                    break
            if last_price:
                return last_price
        # Fallback to history
        data = t.history(period='1d', interval='1d')
        if not data.empty:
            return float(data['Close'].iloc[-1])
        return np.nan
    except Exception:
        return np.nan

@st.cache_data(ttl=1200, show_spinner=False)
def fetch_fundamentals(symbol: str) -> dict:
    sym = symbol if symbol.endswith('.NS') else symbol + '.NS'
    pe = None
    dy = None
    try:
        t = yf.Ticker(sym)
        fi = getattr(t, 'fast_info', None)
        if fi is not None:
            for key in ['trailing_pe', 'trailingPe']:
                val = getattr(fi, key, None) if hasattr(fi, key) else (fi.get(key) if hasattr(fi, 'get') else None)
                if val:
                    pe = float(val)
                    break
            for key in ['dividend_yield', 'dividendYield']:
                val = getattr(fi, key, None) if hasattr(fi, key) else (fi.get(key) if hasattr(fi, 'get') else None)
                if val is not None and val != 0:
                    dy = float(val)
                    break
        if pe is None or dy is None:
            try:
                info = t.get_info() if hasattr(t, 'get_info') else t.info
                if pe is None:
                    pe = float(info.get('trailingPE')) if info.get('trailingPE') else None
                if dy is None:
                    dy = float(info.get('dividendYield')) if info.get('dividendYield') else None
            except Exception:
                pass
    except Exception:
        pass
    return {'pe': pe, 'dividend_yield': dy}

@st.cache_data(ttl=600, show_spinner=False)
def fetch_history(symbol: str, period: str = '1y') -> pd.DataFrame:
    sym = symbol if symbol.endswith('.NS') else symbol + '.NS'
    try:
        t = yf.Ticker(sym)
        df = t.history(period=period, interval='1d')
        if df is not None and not df.empty:
            df = df.rename(columns=str.title)
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def compute_indicators(hist: pd.DataFrame) -> pd.DataFrame:
    df = hist.copy()
    if df.empty:
        return df
    # Moving averages
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()
    # RSI(14)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    df['RSI'] = 100 - (100 / (1 + rs))
    # MACD (12,26,9)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    # 52-week rolling approximations if period permits
    df['RollingHigh252'] = df['Close'].rolling(252).max()
    df['RollingLow252'] = df['Close'].rolling(252).min()
    return df

def compute_atr(hist: pd.DataFrame, period: int = 14) -> pd.Series:
    if hist is None or hist.empty:
        return pd.Series(dtype=float)
    df = hist.copy()
    prev_close = df['Close'].shift(1)
    tr = pd.concat([
        (df['High'] - df['Low']).abs(),
        (df['High'] - prev_close).abs(),
        (df['Low'] - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    return atr

def line_chart_safe(df: pd.DataFrame, cols: list, title: str, window: int = 260):
    st.caption(title)
    if df is None or df.empty:
        st.info('Not enough data to display chart.')
        return
    cols_present = [c for c in cols if c in df.columns]
    if not cols_present:
        st.info('Not enough data to display chart.')
        return
    sub = df[cols_present].dropna(how='all')
    if window:
        sub = sub.tail(window)
    # Keep only columns with at least one non-NaN value
    non_empty_cols = [c for c in sub.columns if sub[c].notna().any()]
    if sub.empty or not non_empty_cols:
        st.info('Not enough data to display chart.')
        return
    st.line_chart(sub[non_empty_cols])

# Fetch live prices
if refresh:
    fetch_live_price.clear()
    fetch_history.clear()
    fetch_fundamentals.clear()

with st.spinner('Fetching live prices...'):
    portfolio_df['live_price'] = portfolio_df['symbol'].apply(fetch_live_price)

# Derived metrics
portfolio_df['live_price_num'] = pd.to_numeric(portfolio_df['live_price'], errors='coerce')
portfolio_df['current_value'] = (portfolio_df['live_price_num'] * portfolio_df['quantity']).fillna(0.0)
portfolio_df['invested_value'] = (portfolio_df['buy_price'] * portfolio_df['quantity']).fillna(0.0)
portfolio_df['profit_loss'] = portfolio_df['current_value'] - portfolio_df['invested_value']
portfolio_df['return_%'] = np.where(portfolio_df['invested_value'] > 0, (portfolio_df['profit_loss'] / portfolio_df['invested_value']) * 100, np.nan)

# KPIs
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric('💰 Total Invested (₹)', f"{portfolio_df['invested_value'].sum():,.2f}")
with col2:
    delta_val = portfolio_df['current_value'].sum() - portfolio_df['invested_value'].sum()
    st.metric('📈 Current Value (₹)', f"{portfolio_df['current_value'].sum():,.2f}", delta=f"{delta_val:,.2f}")
with col3:
    total_ret = delta_val
    total_ret_pct = (total_ret / portfolio_df['invested_value'].sum())*100 if portfolio_df['invested_value'].sum() else 0.0
    st.metric('🔁 Total P/L (₹)', f"{total_ret:,.2f}", delta=f"{total_ret_pct:.2f}%")
with col4:
    winners = (portfolio_df['profit_loss'] > 0).sum()
    st.metric('🗂️ Positions', f"{len(portfolio_df)}", delta=f"{winners} gainers")

st.subheader('Holdings')
st.dataframe(
    portfolio_df[['symbol','quantity','buy_price','live_price_num','current_value','invested_value','profit_loss','return_%']]
        .rename(columns={'live_price_num':'live_price'}),
    use_container_width=True
)

st.subheader('🧭 Signals')

def generate_signal(symbol: str, price: float, buy_price: float):
    hist = fetch_history(symbol, period=analysis_period)
    ind = compute_indicators(hist)
    fundamentals = fetch_fundamentals(symbol)

    score = 0
    reasons = []

    if ind is not None and not ind.empty:
        valid = ind.dropna(subset=['Close'])
        if not valid.empty:
            last = valid.iloc[-1]
            close = float(last['Close'])
            sma50 = last.get('SMA50', np.nan)
            sma200 = last.get('SMA200', np.nan)
            rsi = last.get('RSI', np.nan)
            macd = last.get('MACD', np.nan)
            macd_sig = last.get('MACD_signal', np.nan)

            # Trend (SMA)
            if not np.isnan(sma50) and not np.isnan(sma200):
                if close > sma50 > sma200:
                    score += w_trend
                    reasons.append('Uptrend: Price > SMA50 > SMA200')
                elif close < sma200:
                    score -= max(1, w_trend)
                    reasons.append('Downtrend: Price < SMA200')

            # Momentum (RSI/MACD)
            if not np.isnan(rsi):
                if 45 <= rsi <= 65:
                    score += max(1, w_momentum-1)
                    reasons.append(f'RSI neutral-strong ({rsi:.1f})')
                elif rsi > 70:
                    score -= max(1, w_momentum)
                    reasons.append(f'Overbought RSI ({rsi:.1f})')
                elif rsi < 35:
                    score -= 1
                    reasons.append(f'Weak momentum RSI ({rsi:.1f})')
            if not np.isnan(macd) and not np.isnan(macd_sig):
                if macd > macd_sig:
                    score += 1
                    reasons.append('MACD above signal')
                else:
                    score -= 1
                    reasons.append('MACD below signal')

            # Breakout proximity (60 trading days)
            lookback = ind['Close'].tail(60)
            if len(lookback) >= 20:
                recent_high = lookback.max()
                recent_low = lookback.min()
                if close >= 0.98 * recent_high:
                    score += w_breakout
                    reasons.append('Near recent high (potential breakout)')
                if close <= 1.02 * recent_low:
                    score -= 1
                    reasons.append('Near recent low (weakness)')

            # Risk relative to buy price
            if not pd.isna(buy_price) and buy_price > 0:
                dd = (close - buy_price) / buy_price * 100
                if dd < -10:
                    score -= w_risk
                    reasons.append('Breach -10% from buy (risk management)')

    # Fundamentals (PE, Dividend Yield)
    pe = fundamentals.get('pe')
    dy = fundamentals.get('dividend_yield')
    if pe is not None:
        if pe < 25:
            score += w_value
            reasons.append(f'PE reasonable ({pe:.1f})')
        elif pe > 35:
            score -= w_value
            reasons.append(f'PE rich ({pe:.1f})')
    if dy is not None and dy > 0:
        dy_pct = dy*100 if dy < 1 else dy
        if dy_pct >= 1.0:
            score += 1
            reasons.append(f'Dividend yield {dy_pct:.1f}%')

    # Final recommendation
    if score >= 3:
        rec = 'Buy'
    elif score <= -3:
        rec = 'Exit'
    else:
        rec = 'Hold'
    return rec, int(score), '; '.join(reasons[:6])

sig_rows = []
with st.spinner('Analyzing signals...'):
    for _, r in portfolio_df.iterrows():
        rec, score, reasons = generate_signal(r['symbol'], r['live_price_num'], r['buy_price'])
        sig_rows.append({'symbol': r['symbol'], 'signal': rec, 'score': score, 'reasons': reasons, 'return_%': r['return_%']})
signals_df = pd.DataFrame(sig_rows).sort_values(['signal','score','return_%'], ascending=[True, False, True])

def style_signal(row):
    label = row['signal']
    if label == 'Buy':
        return f"<span class='badge badge-buy'>Buy</span>"
    if label == 'Exit':
        return f"<span class='badge badge-exit'>Exit</span>"
    return f"<span class='badge badge-hold'>Hold</span>"

styled = signals_df.copy()
styled['signal'] = styled.apply(style_signal, axis=1)
styled = styled.rename(columns={'return_%':'return %'})
st.write(styled.to_html(escape=False, index=False), unsafe_allow_html=True)
st.download_button('Download signals CSV', data=signals_df.to_csv(index=False), file_name='signals.csv', mime='text/csv')

# Symbol details
st.subheader('🔍 Symbol detail')
sym = st.selectbox('Choose a symbol to view details', options=portfolio_df['symbol'].tolist())
if sym:
    with st.spinner('Loading history and indicators...'):
        h = fetch_history(sym, period=analysis_period)
        ind = compute_indicators(h)
    if ind is None or ind.empty:
        st.warning('No history available for this symbol.')
    else:
        line_chart_safe(ind, ['Close','SMA20','SMA50','SMA200'], 'Price with SMA20/50/200')
        line_chart_safe(ind, ['RSI'], 'RSI (14)')
        line_chart_safe(ind, ['MACD','MACD_signal'], 'MACD (12,26,9)')

st.info('Educational signals only, not financial advice. Validate with your own research and risk management.')

# --- Broker integration (mStock) ---
def get_mstock_client():
    if not m_base_url or not m_token:
        return None
    cfg = MStockConfig(base_url=m_base_url, access_token=m_token, client_id=m_client_id or None, dry_run=m_dry_run)
    return MStockClient(cfg)

def append_order_log(row: dict, file_path: str = 'orders_log.csv'):
    df = pd.DataFrame([row])
    if os.path.exists(file_path):
        df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        df.to_csv(file_path, index=False)

def read_orders_log(file_path: str = 'orders_log.csv') -> pd.DataFrame:
    if os.path.exists(file_path):
        try:
            return pd.read_csv(file_path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

if m_connect:
    cli = get_mstock_client()
    if cli is None:
        st.warning('Provide Base URL and Access Token to connect.')
    else:
        with st.spinner('Connecting to mStock...'):
            try:
                prof = cli.get_profile()
                st.success('Connected (dry-run)' if m_dry_run else 'Connected')
                st.json(prof)
            except Exception as e:
                st.error(f'Connection failed: {e}')

st.subheader('🧾 Orders (local log)')
orders_df = read_orders_log()
if orders_df.empty:
    st.info('No orders in local log yet.')
else:
    st.dataframe(orders_df, use_container_width=True)
    st.download_button('Download orders log', data=orders_df.to_csv(index=False), file_name='orders_log.csv', mime='text/csv')

with st.expander('Place order via mStock (safe: dry-run by default)'):
    order_symbol = st.selectbox('Symbol', options=portfolio_df['symbol'].tolist())
    order_side = st.radio('Side', options=['BUY','SELL'], horizontal=True)
    order_qty = st.number_input('Quantity', min_value=1, step=1, value=int(max(1, portfolio_df.loc[portfolio_df['symbol']==order_symbol, 'quantity'].max() or 1)))
    order_type = st.selectbox('Order Type', options=['MARKET','LIMIT'])
    order_product = st.selectbox('Product', options=['CNC','MIS','NRML'])
    order_price = st.number_input('Limit Price (if LIMIT)', min_value=0.0, step=0.05, value=0.0, format='%0.2f')
    order_go = st.button('Place order')
    if order_go:
        cli = get_mstock_client()
        if cli is None:
            st.warning('Configure Base URL and Access Token in the sidebar to place orders.')
        else:
            with st.spinner('Sending order...'):
                try:
                    payload_price = float(order_price) if order_type == 'LIMIT' and order_price > 0 else None
                    resp = cli.place_order(
                        symbol=order_symbol,
                        side=order_side,
                        quantity=int(order_qty),
                        order_type=order_type,
                        product=order_product,
                        price=payload_price,
                        remarks='streamlit-portfolio'
                    )
                    ts = datetime.now().isoformat(timespec='seconds')
                    log_row = {
                        'ts': ts,
                        'symbol': order_symbol,
                        'side': order_side,
                        'quantity': int(order_qty),
                        'order_type': order_type,
                        'product': order_product,
                        'price': payload_price,
                        'dry_run': m_dry_run,
                        'response': str(resp)[:500],
                    }
                    append_order_log(log_row)
                    st.success('Order sent (dry-run)' if m_dry_run else 'Order placed.')
                except Exception as e:
                    st.error(f'Order failed: {e}')

# --- Research a new stock ---
st.subheader('🧪 Research a new stock for potential addition')
if research_symbol_input:
    symbol_raw = research_symbol_input.strip().upper()
else:
    symbol_raw = ''

if research_go and symbol_raw:
    # Normalize to base symbol without suffix; fetchers add .NS
    base_symbol = symbol_raw[:-3] if symbol_raw.endswith('.NS') else symbol_raw
    # Choose period by horizon
    research_period = '6mo' if research_horizon == 'Short-term' else '5y'
    with st.spinner('Analyzing candidate...'):
        hist_r = fetch_history(base_symbol, period=research_period)
        ind_r = compute_indicators(hist_r)
        fund_r = fetch_fundamentals(base_symbol)
        atr14 = compute_atr(hist_r, 14)

    if ind_r is None or ind_r.empty:
        st.warning('No data available for this symbol. Please check the ticker or try another one.')
    else:
        last_row = ind_r.dropna(subset=['Close']).iloc[-1] if not ind_r.dropna(subset=['Close']).empty else ind_r.iloc[-1]
        close = float(last_row['Close'])
        sma20 = last_row.get('SMA20', np.nan)
        sma50 = last_row.get('SMA50', np.nan)
        sma200 = last_row.get('SMA200', np.nan)
        rsi = last_row.get('RSI', np.nan)
        macd = last_row.get('MACD', np.nan)
        macd_sig = last_row.get('MACD_signal', np.nan)
        atr_last = float(atr14.dropna().iloc[-1]) if not atr14.dropna().empty else np.nan

        # Build horizon-specific score
        score = 0
        reasons = []
        lookback = ind_r['Close'].tail(60)
        if research_horizon == 'Short-term':
            if not np.isnan(sma20) and not np.isnan(sma50) and close > sma20 > sma50:
                score += 2; reasons.append('Uptrend ST: Price > SMA20 > SMA50')
            if not np.isnan(sma50) and close < sma50:
                score -= 1; reasons.append('Below SMA50 (weak ST trend)')
            if not np.isnan(macd) and not np.isnan(macd_sig) and macd > macd_sig:
                score += 1; reasons.append('MACD > signal')
            if not np.isnan(rsi):
                if 45 <= rsi <= 65:
                    score += 1; reasons.append(f'RSI supportive ({rsi:.1f})')
                elif rsi > 70:
                    score -= 1; reasons.append(f'Overbought RSI ({rsi:.1f})')
                elif rsi < 35:
                    score -= 1; reasons.append(f'Weak RSI ({rsi:.1f})')
            if len(lookback) >= 20 and close >= 0.99 * lookback.max():
                score += 1; reasons.append('Near recent high (breakout potential)')
        else:  # Long-term
            if not np.isnan(sma50) and not np.isnan(sma200) and sma50 > sma200:
                score += 2; reasons.append('Golden trend: SMA50 > SMA200')
            if not np.isnan(sma200) and close > sma200:
                score += 1; reasons.append('Price above SMA200')
            if not np.isnan(macd) and not np.isnan(macd_sig) and macd > macd_sig:
                score += 1; reasons.append('MACD > signal')
            if not np.isnan(rsi) and rsi > 70:
                score -= 1; reasons.append('Overbought RSI')
            pe = fund_r.get('pe')
            if pe is not None:
                if pe < 25:
                    score += 1; reasons.append(f'PE reasonable ({pe:.1f})')
                elif pe > 35:
                    score -= 1; reasons.append(f'PE rich ({pe:.1f})')
            dy = fund_r.get('dividend_yield')
            if dy is not None and dy > 0:
                dy_pct = dy*100 if dy < 1 else dy
                if dy_pct >= 1.0:
                    score += 1; reasons.append(f'Dividend yield {dy_pct:.1f}%')

        # Recommendation
        if score >= 3:
            rec = 'Consider Add'
        elif score <= -2:
            rec = 'Avoid'
        else:
            rec = 'Watchlist'

        # Entry/Stop suggestions
        if research_horizon == 'Short-term':
            entry_ref = np.nanmax([sma20, sma50]) if not (np.isnan(sma20) and np.isnan(sma50)) else close
            entry = max(entry_ref, float(lookback.max()) if len(lookback) > 0 else close)
            stop = (entry - 1.5*atr_last) if not np.isnan(atr_last) else (entry * 0.95)
        else:
            entry = max(sma200 if not np.isnan(sma200) else close, close)
            stop = (entry - 2.5*atr_last) if not np.isnan(atr_last) else (entry * 0.92)
        risk_pct = ((entry - stop) / entry * 100) if entry and stop else np.nan

        # Display summary
        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Last Price (₹)', f'{close:,.2f}')
        c2.metric('RSI(14)', f'{rsi:.1f}' if not np.isnan(rsi) else 'N/A')
        c3.metric('ATR(14)', f'{atr_last:,.2f}' if not np.isnan(atr_last) else 'N/A')
        c4.metric('PE', f"{fund_r.get('pe'):.1f}" if fund_r.get('pe') is not None else 'N/A')
        c1.metric('SMA50', f'{sma50:,.2f}' if not np.isnan(sma50) else 'N/A')
        c2.metric('SMA200', f'{sma200:,.2f}' if not np.isnan(sma200) else 'N/A')
        dy_val = fund_r.get('dividend_yield')
        dy_pct_disp = (dy_val*100 if (dy_val is not None and dy_val < 1) else dy_val)
        c3.metric('Dividend Yield', f"{dy_pct_disp:.2f}%" if dy_pct_disp is not None else 'N/A')
        c4.metric('Recommendation', rec, delta=f'Score {score}')

        st.caption('Entry/Stop suggestions (illustrative)')
        st.write(f"Suggested entry: ₹{entry:,.2f} | Suggested stop: ₹{stop:,.2f} | Risk per share: {risk_pct:.2f}%")
        st.write('Reasons:')
        st.markdown('\n'.join([f'- {r}' for r in reasons]) or '- No strong signals detected')

        # Charts
    line_chart_safe(ind_r, ['Close','SMA20','SMA50','SMA200'], 'Price with SMA20/50/200')
    line_chart_safe(ind_r, ['RSI'], 'RSI (14)')
    line_chart_safe(ind_r, ['MACD','MACD_signal'], 'MACD (12,26,9)')
