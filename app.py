import streamlit as st
import pandas as pd
import yfinance as yf
from lightweight_charts_v5 import lightweight_charts_v5_component
import time
from datetime import datetime, timezone

# ============================================================================
# SESSION STATE - PAPER TRADING & POSITIONS
# ============================================================================
if 'balance' not in st.session_state:
    st.session_state.balance = 100000.0
if 'starting_balance' not in st.session_state:
    st.session_state.starting_balance = 100000.0
if 'positions' not in st.session_state:
    st.session_state.positions = []
if 'trade_history' not in st.session_state:
    st.session_state.trade_history = []
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()

# ============================================================================
# STYLING
# ============================================================================
st.set_page_config(layout="wide", page_title="Pro Trading Terminal", page_icon="📊")
st.markdown("""
    <style>
    .main { background-color: #0a0c0f !important; color: #e0e3e7; }
    div[data-testid="stMetric"] { 
        background-color: #161b22; 
        border: 1px solid #30363d; 
        border-radius: 8px; 
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    .pattern-card-bullish { 
        background: linear-gradient(135deg, #1a3a2e 0%, #16342a 100%);
        padding: 18px; 
        border-radius: 10px; 
        border-left: 4px solid #26a69a;
        margin: 8px 0;
        box-shadow: 0 3px 6px rgba(0,0,0,0.4);
    }
    .pattern-card-bearish { 
        background: linear-gradient(135deg, #3a1a1f 0%, #341619 100%);
        padding: 18px; 
        border-radius: 10px; 
        border-left: 4px solid #ef5350;
        margin: 8px 0;
        box-shadow: 0 3px 6px rgba(0,0,0,0.4);
    }
    .pattern-card-neutral { 
        background: linear-gradient(135deg, #3a3520 0%, #342f1a 100%);
        padding: 18px; 
        border-radius: 10px; 
        border-left: 4px solid #fbbf24;
        margin: 8px 0;
        box-shadow: 0 3px 6px rgba(0,0,0,0.4);
    }
    .stButton>button {
        background: linear-gradient(135deg, #26a69a 0%, #1f8b7f 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #1f8b7f 0%, #186d64 100%);
        box-shadow: 0 4px 12px rgba(38, 166, 154, 0.4);
    }
    h1, h2, h3 { color: #e0e3e7 !important; }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA FETCHING - FIXED TIMESTAMPS
# ============================================================================
@st.cache_data(ttl=30)
def fetch_market_data(ticker, tf):
    """Fetch with PROPER timestamp handling"""
    period_map = {"1m": "1d", "5m": "5d", "15m": "5d", "1h": "1mo", "1d": "6mo"}
    
    try:
        df = yf.download(ticker, period=period_map.get(tf, "5d"), interval=tf, progress=False)
        if df.empty: return pd.DataFrame()
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df = df.reset_index()
        ts_col = 'Datetime' if 'Datetime' in df.columns else 'Date'
        
        # CRITICAL TIMESTAMP FIX
        if tf in ['1m', '5m', '15m', '1h']:
            if df[ts_col].dt.tz is None:
                df[ts_col] = pd.to_datetime(df[ts_col]).dt.tz_localize('America/New_York')
            df[ts_col] = df[ts_col].dt.tz_convert('America/New_York')
        
        # Convert to Unix seconds
        df['time'] = (df[ts_col].astype(int) / 10**9).astype(int)
        df['datetime'] = df[ts_col]
        
        df = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
        df = df.dropna(subset=['open', 'high', 'low', 'close'])
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        return df
    except Exception as e:
        st.error(f"Data fetch error: {e}")
        return pd.DataFrame()

# ============================================================================
# PATTERN DETECTION (Top 5 only)
# ============================================================================
def detect_all_patterns(df):
    """Detect patterns and return top 5 by confidence"""
    if len(df) < 20:
        return [], [], "SCANNING"
    
    markers, signals = [], []
    recent = df.tail(50).reset_index(drop=True)
    
    # Fair Value Gap
    for i in range(2, len(recent)):
        bullish_gap = recent['low'].iloc[i] - recent['high'].iloc[i-2]
        if bullish_gap > 0 and (bullish_gap / recent['close'].iloc[i] * 100) > 0.05:
            signals.append({
                "name": "Bullish FVG",
                "category": "ICT",
                "type": "bullish",
                "confidence": 85,
                "price": recent['close'].iloc[i],
                "logic": f"Buy imbalance. Gap: ${bullish_gap:.2f}"
            })
            markers.append({"time": int(recent['time'].iloc[i]), "position": "belowBar", "color": "#26a69a", "shape": "arrowUp", "text": "FVG"})
        
        bearish_gap = recent['low'].iloc[i-2] - recent['high'].iloc[i]
        if bearish_gap > 0 and (bearish_gap / recent['close'].iloc[i] * 100) > 0.05:
            signals.append({
                "name": "Bearish FVG",
                "category": "ICT",
                "type": "bearish",
                "confidence": 85,
                "price": recent['close'].iloc[i],
                "logic": f"Sell pressure. Gap: ${bearish_gap:.2f}"
            })
            markers.append({"time": int(recent['time'].iloc[i]), "position": "aboveBar", "color": "#ef5350", "shape": "arrowDown", "text": "FVG"})
    
    # Displacement
    for i in range(5, len(recent)):
        body = abs(recent['close'].iloc[i] - recent['open'].iloc[i])
        avg_body = sum([abs(recent['close'].iloc[j] - recent['open'].iloc[j]) for j in range(i-5, i)]) / 5
        if avg_body > 0 and body > avg_body * 2.5:
            is_bull = recent['close'].iloc[i] > recent['open'].iloc[i]
            signals.append({
                "name": f"{'Bullish' if is_bull else 'Bearish'} Displacement",
                "category": "ICT",
                "type": "bullish" if is_bull else "bearish",
                "confidence": 82,
                "price": recent['close'].iloc[i],
                "logic": f"Momentum {body/avg_body:.1f}x larger"
            })
            markers.append({"time": int(recent['time'].iloc[i]), "position": "belowBar" if is_bull else "aboveBar", "color": "#fbbf24", "shape": "text", "text": "⚡"})
    
    # Engulfing
    for i in range(1, len(recent)):
        prev, curr = recent.iloc[i-1], recent.iloc[i]
        if prev['close'] < prev['open'] and curr['close'] > curr['open']:
            if curr['close'] > prev['open'] and curr['open'] < prev['close']:
                signals.append({
                    "name": "Bullish Engulfing",
                    "category": "Candlestick",
                    "type": "bullish",
                    "confidence": 78,
                    "price": curr['close'],
                    "logic": "Buyers overwhelmed sellers"
                })
                markers.append({"time": int(curr['time']), "position": "belowBar", "color": "#26a69a", "shape": "circle", "text": "BE"})
        
        if prev['close'] > prev['open'] and curr['close'] < curr['open']:
            if curr['close'] < prev['open'] and curr['open'] > prev['close']:
                signals.append({
                    "name": "Bearish Engulfing",
                    "category": "Candlestick",
                    "type": "bearish",
                    "confidence": 78,
                    "price": curr['close'],
                    "logic": "Sellers overwhelmed buyers"
                })
                markers.append({"time": int(curr['time']), "position": "aboveBar", "color": "#ef5350", "shape": "circle", "text": "BE"})
    
    # Three Outside Down/Up
    for i in range(2, len(recent)):
        first, second, third = recent.iloc[i-2], recent.iloc[i-1], recent.iloc[i]
        if (first['close'] > first['open'] and second['close'] < second['open'] and
            second['close'] < first['open'] and second['open'] > first['close'] and
            third['close'] < second['close']):
            signals.append({
                "name": "Three Outside Down",
                "category": "Candlestick",
                "type": "bearish",
                "confidence": 92,
                "price": third['close'],
                "logic": "HIGH PROB bearish continuation"
            })
            markers.append({"time": int(third['time']), "position": "aboveBar", "color": "#ef5350", "shape": "circle", "text": "3↓"})
        
        if (first['close'] < first['open'] and second['close'] > second['open'] and
            second['close'] > first['open'] and second['open'] < first['close'] and
            third['close'] > second['close']):
            signals.append({
                "name": "Three Outside Up",
                "category": "Candlestick",
                "type": "bullish",
                "confidence": 92,
                "price": third['close'],
                "logic": "HIGH PROB bullish continuation"
            })
            markers.append({"time": int(third['time']), "position": "belowBar", "color": "#26a69a", "shape": "circle", "text": "3↑"})
    
    # Hammer
    for i in range(5, len(recent)):
        c = recent.iloc[i]
        body = abs(c['close'] - c['open'])
        lower_wick = min(c['open'], c['close']) - c['low']
        upper_wick = c['high'] - max(c['open'], c['close'])
        if lower_wick > body * 2 and upper_wick < body * 0.5 and body > 0:
            is_downtrend = recent['close'].iloc[i-5] > recent['close'].iloc[i-1]
            if is_downtrend:
                signals.append({
                    "name": "Hammer",
                    "category": "Candlestick",
                    "type": "bullish",
                    "confidence": 75,
                    "price": c['close'],
                    "logic": "Bullish reversal. Buyers rejected lows"
                })
                markers.append({"time": int(c['time']), "position": "belowBar", "color": "#26a69a", "shape": "text", "text": "🔨"})
    
    # Double Top/Bottom
    if len(recent) >= 15:
        highs = recent['high'].values
        peaks = []
        for i in range(2, len(recent)-2):
            if highs[i] == max(highs[max(0, i-2):min(len(highs), i+3)]):
                peaks.append((i, highs[i]))
        
        if len(peaks) >= 2:
            last_two = peaks[-2:]
            if abs(last_two[0][1] - last_two[1][1]) / last_two[0][1] < 0.02:
                signals.append({
                    "name": "Double Top",
                    "category": "Chart Pattern",
                    "type": "bearish",
                    "confidence": 75,
                    "price": recent['close'].iloc[-1],
                    "logic": f"Resistance at ${last_two[0][1]:.2f}"
                })
    
    # Sort by confidence, return top 5
    signals_sorted = sorted(signals, key=lambda x: x['confidence'], reverse=True)[:5]
    
    if signals_sorted:
        bullish_count = sum(1 for s in signals_sorted if s['type'] == 'bullish')
        bearish_count = sum(1 for s in signals_sorted if s['type'] == 'bearish')
        sentiment = "BULLISH" if bullish_count > bearish_count else "BEARISH" if bearish_count > bullish_count else "NEUTRAL"
    else:
        sentiment = "NEUTRAL"
    
    return markers, signals_sorted, sentiment

# ============================================================================
# PAPER TRADING FUNCTIONS
# ============================================================================
def execute_paper_trade(symbol, qty, current_price, trade_type, tp=None, sl=None):
    """Execute a paper trade"""
    cost = qty * current_price
    
    if trade_type == "BUY":
        if st.session_state.balance < cost:
            return False, "❌ Insufficient balance"
        
        st.session_state.balance -= cost
        st.session_state.positions.append({
            "symbol": symbol,
            "qty": qty,
            "entry_price": current_price,
            "tp": tp,
            "sl": sl,
            "type": "LONG",
            "open_time": datetime.now()
        })
        return True, f"✅ Bought {qty} shares at ${current_price:.2f}"
    
    elif trade_type == "SELL":
        for pos in st.session_state.positions:
            if pos['symbol'] == symbol and pos['type'] == "LONG" and pos['qty'] >= qty:
                profit = (current_price - pos['entry_price']) * qty
                st.session_state.balance += (pos['entry_price'] * qty) + profit
                
                st.session_state.trade_history.append({
                    "symbol": symbol,
                    "qty": qty,
                    "entry": pos['entry_price'],
                    "exit": current_price,
                    "profit": profit,
                    "profit_pct": (profit / (pos['entry_price'] * qty)) * 100,
                    "close_time": datetime.now()
                })
                
                pos['qty'] -= qty
                if pos['qty'] == 0:
                    st.session_state.positions.remove(pos)
                
                return True, f"✅ Sold {qty} shares at ${current_price:.2f}. P/L: ${profit:+.2f}"
        
        return False, "❌ No open position to sell"

def check_tp_sl(symbol, current_price):
    """Check if any positions hit TP/SL"""
    for pos in st.session_state.positions[:]:
        if pos['symbol'] != symbol or pos['type'] != "LONG":
            continue
        
        # Check Stop Loss
        if pos['sl'] and current_price <= pos['sl']:
            profit = (pos['sl'] - pos['entry_price']) * pos['qty']
            st.session_state.balance += pos['sl'] * pos['qty']
            st.session_state.trade_history.append({
                "symbol": pos['symbol'],
                "qty": pos['qty'],
                "entry": pos['entry_price'],
                "exit": pos['sl'],
                "profit": profit,
                "profit_pct": (profit / (pos['entry_price'] * pos['qty'])) * 100,
                "close_time": datetime.now(),
                "reason": "Stop Loss"
            })
            st.session_state.positions.remove(pos)
            st.toast(f"🛑 Stop Loss hit at ${pos['sl']:.2f}")
        
        # Check Take Profit
        elif pos['tp'] and current_price >= pos['tp']:
            profit = (pos['tp'] - pos['entry_price']) * pos['qty']
            st.session_state.balance += pos['tp'] * pos['qty']
            st.session_state.trade_history.append({
                "symbol": pos['symbol'],
                "qty": pos['qty'],
                "entry": pos['entry_price'],
                "exit": pos['tp'],
                "profit": profit,
                "profit_pct": (profit / (pos['entry_price'] * pos['qty'])) * 100,
                "close_time": datetime.now(),
                "reason": "Take Profit"
            })
            st.session_state.positions.remove(pos)
            st.toast(f"🎯 Take Profit hit at ${pos['tp']:.2f}")

# ============================================================================
# SIDEBAR
# ============================================================================
st.sidebar.image("unnamed.jpg", use_container_width=True)
st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.title("Trading Terminal")

symbol = st.sidebar.text_input("Symbol", value="SPY").upper()
timeframe = st.sidebar.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "1d"], index=1)

st.sidebar.markdown("---")
st.sidebar.subheader("🔄 Data Refresh")
auto_refresh = st.sidebar.checkbox("Auto-Refresh (30s)", value=False)

if st.sidebar.button("🔄 Refresh Now", use_container_width=True):
    st.cache_data.clear()
    st.session_state.last_refresh = time.time()
    st.rerun()

st.sidebar.markdown("---")

# ============================================================================
# PAPER TRADING PANEL
# ============================================================================
st.sidebar.subheader("💼 Paper Trading")
pnl = st.session_state.balance - st.session_state.starting_balance
pnl_pct = (pnl / st.session_state.starting_balance) * 100

st.sidebar.metric("Account Balance", f"${st.session_state.balance:,.2f}", f"{pnl:+,.2f} ({pnl_pct:+.2f}%)")

# Trading Form
with st.sidebar.form("trade_form"):
    st.write("**Place Order**")
    trade_action = st.radio("Action", ["BUY", "SELL"], horizontal=True)
    trade_qty = st.number_input("Quantity", min_value=1, value=10, step=1)
    
    col1, col2 = st.columns(2)
    with col1:
        use_tp = st.checkbox("Take Profit")
    with col2:
        use_sl = st.checkbox("Stop Loss")
    
    tp_price = None
    sl_price = None
    
    if use_tp:
        tp_price = st.number_input("TP Price $", min_value=0.01, value=500.0, step=0.01, format="%.2f")
    if use_sl:
        sl_price = st.number_input("SL Price $", min_value=0.01, value=450.0, step=0.01, format="%.2f")
    
    submitted = st.form_submit_button("📊 Execute Trade", use_container_width=True)

# Open Positions
if st.session_state.positions:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Open Positions")
    
    for pos in st.session_state.positions:
        current_value = pos['qty'] * pos['entry_price']
        with st.sidebar.expander(f"{pos['symbol']} ({pos['qty']} shares)"):
            st.write(f"**Entry:** ${pos['entry_price']:.2f}")
            st.write(f"**Value:** ${current_value:,.2f}")
            if pos['tp']:
                st.write(f"**TP:** ${pos['tp']:.2f} 🎯")
            if pos['sl']:
                st.write(f"**SL:** ${pos['sl']:.2f} 🛑")
            st.caption(f"Opened: {pos['open_time'].strftime('%H:%M:%S')}")

# Reset Account
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reset Account", use_container_width=True):
    st.session_state.balance = 100000.0
    st.session_state.starting_balance = 100000.0
    st.session_state.positions = []
    st.session_state.trade_history = []
    st.sidebar.success("✅ Account reset to $100,000")
    time.sleep(1)
    st.rerun()

# ============================================================================
# MAIN DASHBOARD
# ============================================================================
header_col1, header_col2 = st.columns([1, 6]) # Adjust ratio to fit your preference

with header_col1:
    st.image("unnamed.jpg", width=70) # Set a fixed width for the header icon

with header_col2:
    st.title("R-Quant AI Terminal")
    st.caption("Built by Rajan. Powered by Data. Driven by ICT." "Your Sovereign Command Center for the Global Markets.")

df = fetch_market_data(symbol, timeframe)

if not df.empty:
    current_price = df['close'].iloc[-1]
    prev_price = df['close'].iloc[-2] if len(df) > 1 else current_price
    price_change = ((current_price - prev_price) / prev_price) * 100
    
    # Check TP/SL
    check_tp_sl(symbol, current_price)
    
    # Handle trade submission
    if submitted:
        success, message = execute_paper_trade(symbol, trade_qty, current_price, trade_action, tp_price, sl_price)
        if success:
            st.sidebar.success(message)
            time.sleep(0.5)
            st.rerun()
        else:
            st.sidebar.error(message)
    
    # Detect patterns
    markers, top_patterns, sentiment = detect_all_patterns(df)
    
    # Metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric(symbol, f"${current_price:.2f}", f"{price_change:+.2f}%")
    m2.metric("Bias", sentiment, "🟢" if sentiment == "BULLISH" else "🔴" if sentiment == "BEARISH" else "⚪")
    m3.metric("Patterns", len(top_patterns))
    m4.metric("Account P/L", f"${pnl:+,.0f}", f"{pnl_pct:+.2f}%")
    m5.metric("Open Positions", len(st.session_state.positions))
    
    # Chart
    st.subheader(f"📈 {symbol} Chart - {timeframe.upper()}")
    
    candles = df[['time', 'open', 'high', 'low', 'close']].to_dict('records')
    volumes = [{"time": int(r['time']), "value": float(r['volume']), 
                "color": '#26a69a' if r['close'] >= r['open'] else '#ef5350'} for _, r in df.iterrows()]
    
    chart_opts = {
        "layout": {"background": {"color": "#0a0c0f"}, "textColor": "#e0e3e7"},
        "grid": {"vertLines": {"color": "#1a1d28"}, "horzLines": {"color": "#1a1d28"}},
        "timeScale": {"timeVisible": True, "secondsVisible": timeframe in ['1m', '5m'], "borderColor": "#30363d"},
        "crosshair": {
            "mode": 0,
            "vertLine": {"width": 1, "color": "rgba(224, 227, 235, 0.1)", "labelBackgroundColor": "#2962FF"},
            "horzLine": {"width": 1, "color": "rgba(224, 227, 235, 0.1)", "labelBackgroundColor": "#2962FF"}
        }
    }
    
    lightweight_charts_v5_component(
        f"chart_{symbol}_{timeframe}",
        charts=[
            {
                "series": [{
                    "type": "Candlestick",
                    "data": candles,
                    "markers": markers,
                    "options": {
                        "upColor": "#26a69a", "downColor": "#ef5350",
                        "borderUpColor": "#26a69a", "borderDownColor": "#ef5350",
                        "wickUpColor": "#26a69a", "wickDownColor": "#ef5350"
                    }
                }],
                "options": chart_opts,
                "height": 500
            },
            {
                "series": [{"type": "Histogram", "data": volumes, "options": {"priceFormat": {"type": "volume"}}}],
                "options": {**chart_opts, "rightPriceScale": {"scaleMargins": {"top": 0.8, "bottom": 0}}},
                "height": 150
            }
        ],
        height=700,
        key=f"chart_{symbol}_{timeframe}"
    )
    
    # Top 5 Patterns
    st.subheader("🎯 Top 5 Detected Patterns")
    
    if top_patterns:
        cols = st.columns(5)
        for idx, pattern in enumerate(top_patterns):
            with cols[idx]:
                card_class = f"pattern-card-{pattern['type']}"
                emoji = '🟢' if pattern['type'] == 'bullish' else '🔴' if pattern['type'] == 'bearish' else '⚪'
                st.markdown(f"""
                <div class="{card_class}">
                    <div style="font-size:20px; margin-bottom:8px;">{emoji}</div>
                    <h4 style="margin:0; font-size:15px; font-weight:700;">{pattern['name']}</h4>
                    <p style="margin:5px 0; font-size:11px; opacity:0.7;">{pattern['category']}</p>
                    <p style="margin:10px 0; font-size:12px; line-height:1.4;">{pattern['logic']}</p>
                    <div style="margin-top:12px; padding-top:12px; border-top:1px solid rgba(255,255,255,0.1);">
                        <div style="font-size:13px; font-weight:700;">Confidence: {pattern['confidence']}%</div>
                        <div style="margin-top:4px; font-size:11px; opacity:0.7;">Price: ${pattern['price']:.2f}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("📊 No patterns detected. Market consolidating.")
    
    # Trade History
    if st.session_state.trade_history:
        st.subheader("📜 Trade History")
        with st.expander("View All Trades"):
            history_df = pd.DataFrame(st.session_state.trade_history)
            history_df['close_time'] = history_df['close_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
            history_df = history_df[['symbol', 'qty', 'entry', 'exit', 'profit', 'profit_pct', 'close_time']]
            history_df.columns = ['Symbol', 'Qty', 'Entry $', 'Exit $', 'Profit $', 'Profit %', 'Time']
            st.dataframe(history_df, use_container_width=True)
            
            # Stats
            total_profit = sum([t['profit'] for t in st.session_state.trade_history])
            winning_trades = len([t for t in st.session_state.trade_history if t['profit'] > 0])
            total_trades = len(st.session_state.trade_history)
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Trades", total_trades)
            col2.metric("Win Rate", f"{win_rate:.1f}%")
            col3.metric("Total P/L", f"${total_profit:+,.2f}")
    
    st.caption(f"🕐 {datetime.now().strftime('%H:%M:%S')} | Yahoo Finance")

else:
    st.error(f"❌ Failed to load {symbol}")

# Auto-refresh
if auto_refresh and (time.time() - st.session_state.last_refresh) > 30:
    st.session_state.last_refresh = time.time()
    st.cache_data.clear()
    st.rerun()