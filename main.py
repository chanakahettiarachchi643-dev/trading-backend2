import os
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ------------------ INDICATOR CALCULATIONS ------------------
def rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    # FIX: avoid division by zero
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))

def macd(series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def bollinger_bands(series, period=20, std=2):
    sma = series.rolling(period).mean()
    std_dev = series.rolling(period).std()
    upper = sma + std * std_dev
    lower = sma - std * std_dev
    return upper, sma, lower

def stochastic(high, low, close, k_period=14, d_period=3):
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    denom = (highest_high - lowest_low).replace(0, 1e-10)  # FIX: avoid division by zero
    k = 100 * ((close - lowest_low) / denom)
    d = k.rolling(d_period).mean()
    return k, d

def atr(high, low, close, period=14):
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def sma(series, period):
    return series.rolling(period).mean()

def bb_squeeze(high, low, close, bb_period=20, keltner_period=20):
    upper_bb, mid_bb, lower_bb = bollinger_bands(close, bb_period)
    atr_val = atr(high, low, close, keltner_period)
    typical_price = (high + low + close) / 3
    keltner_upper = typical_price.rolling(keltner_period).mean() + 1.5 * atr_val
    keltner_lower = typical_price.rolling(keltner_period).mean() - 1.5 * atr_val
    squeeze_on = (upper_bb < keltner_upper) & (lower_bb > keltner_lower)
    return squeeze_on

def volume_spike(volume, period=20):
    avg_vol = volume.rolling(period).mean()
    return volume > 1.5 * avg_vol

# FIX: unified timeframe to minutes converter
def timeframe_to_minutes(timeframe: str) -> int:
    mapping = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240}
    if timeframe in mapping:
        return mapping[timeframe]
    # fallback: try stripping 'm'
    try:
        return int(timeframe.replace("m", ""))
    except:
        return 5  # default 5 minutes

# ------------------ DATA FETCHING ------------------
def get_crypto_candles(symbol, timeframe, limit=100):
    intervals = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h", "4h": "4h"}
    if timeframe not in intervals:
        return None
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={intervals[timeframe]}&limit={limit}"
    try:
        resp = requests.get(url, timeout=10).json()
    except Exception:
        return None
    if "code" in resp:
        return None
    df = pd.DataFrame(resp, columns=[
        "time","open","high","low","close","volume",
        "close_time","quote_vol","trades",
        "taker_buy_base","taker_buy_quote","ignore"
    ])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)
    return df

def get_forex_candles(symbol, timeframe, limit=100):
    np.random.seed(hash(symbol) % 2**32)
    base_price = {"EURUSD": 1.08, "GBPUSD": 1.26, "USDJPY": 144.0, "XAUUSD": 2320}.get(symbol, 1.0)
    now = datetime.utcnow()
    # FIX: use unified converter
    minutes = timeframe_to_minutes(timeframe)
    timestamps = [now - timedelta(minutes=i * minutes) for i in range(limit - 1, -1, -1)]
    prices = [base_price]
    for _ in range(limit - 1):
        prices.append(prices[-1] * (1 + np.random.normal(0, 0.0002)))
    df = pd.DataFrame({
        "time": timestamps,
        "open": prices,
        "high": [p * (1 + abs(np.random.normal(0, 0.0002))) for p in prices],
        "low":  [p * (1 - abs(np.random.normal(0, 0.0002))) for p in prices],
        "close":[p * (1 + np.random.normal(0, 0.0001)) for p in prices],
        "volume": np.random.randint(100, 1000, limit)
    })
    return df

def fetch_data(symbol, market, timeframe):
    if market == "crypto":
        return get_crypto_candles(symbol, timeframe)
    else:
        return get_forex_candles(symbol, timeframe)

# ------------------ CONFLUENCE SCORING ------------------
def compute_confluence(df):
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    last_close = close.iloc[-1]

    score = 0
    details = []

    # RSI
    rsi_val = rsi(close).iloc[-1]
    if pd.isna(rsi_val):
        rsi_val = 50.0
    if rsi_val < 30:
        score += 25
        details.append("RSI oversold")
    elif rsi_val > 70:
        score -= 25
        details.append("RSI overbought")

    # MACD
    macd_line, signal_line, _ = macd(close)
    if macd_line.iloc[-1] > signal_line.iloc[-1] and macd_line.iloc[-2] <= signal_line.iloc[-2]:
        score += 20
        details.append("MACD bullish crossover")
    elif macd_line.iloc[-1] < signal_line.iloc[-1] and macd_line.iloc[-2] >= signal_line.iloc[-2]:
        score -= 20
        details.append("MACD bearish crossover")

    # EMA alignment
    ema50  = ema(close, 50)
    ema100 = ema(close, 100)
    ema200 = ema(close, 200)
    if ema50.iloc[-1] > ema100.iloc[-1] > ema200.iloc[-1]:
        score += 15
        details.append("EMAs bullish aligned")
    elif ema50.iloc[-1] < ema100.iloc[-1] < ema200.iloc[-1]:
        score -= 15
        details.append("EMAs bearish aligned")

    # Bollinger Bands
    upper, mid, lower = bollinger_bands(close)
    if last_close <= lower.iloc[-1]:
        score += 15
        details.append("Price at lower BB")
    elif last_close >= upper.iloc[-1]:
        score -= 15
        details.append("Price at upper BB")

    # Stochastic
    stoch_k, stoch_d = stochastic(high, low, close)
    if stoch_k.iloc[-1] < 20 and stoch_d.iloc[-1] < 20:
        score += 10
        details.append("Stochastic oversold")
    elif stoch_k.iloc[-1] > 80 and stoch_d.iloc[-1] > 80:
        score -= 10
        details.append("Stochastic overbought")

    # ATR
    atr_val = atr(high, low, close).iloc[-1]

    # BB Squeeze
    squeeze = bb_squeeze(high, low, close).iloc[-1]
    if squeeze:
        details.append("BB Squeeze (low volatility)")
        score += 5

    # Volume spike
    if volume_spike(volume).iloc[-1]:
        details.append("Volume spike")
        score += 10

    score = max(-100, min(100, score))
    return score, details, rsi_val, atr_val

# ------------------ SIGNAL GENERATION ------------------
@app.get("/signal")
def generate_signal(
    symbol: str = Query("BTCUSDT"),
    market: str = Query("crypto"),
    platform: str = Query("binance"),
    timeframe: str = Query("5m"),
    amount: float = Query(100.0)
):
    if platform in ["expertoption", "pocketoption"]:
        market = "binary"
    elif platform in ["xm", "deriv", "icmarkets"]:
        market = "forex"
    else:
        market = "crypto"

    df = fetch_data(symbol, market, timeframe)
    if df is None or len(df) < 50:
        return {"error": "Not enough data"}

    score, details, rsi_val, atr_val = compute_confluence(df)
    close = df["close"].iloc[-1]

    if score > 30:
        direction = "BUY"
    elif score < -30:
        direction = "SELL"
    else:
        direction = "WAIT"

    if market == "forex":
        sl_mult, tp_mult = 1.5, 2.5
    elif market == "binary":
        sl_mult, tp_mult = 0.0, 0.0
    else:
        sl_mult, tp_mult = 2.0, 3.0

    if market == "binary":
        # FIX: binary options - show expiry info instead of SL/TP
        sl = None
        tp = None
        expiry_minutes = timeframe_to_minutes(timeframe)
        expiry_note = f"Expiry: {expiry_minutes} min"
    else:
        sl = round(close - atr_val * sl_mult, 5) if direction == "BUY" else round(close + atr_val * sl_mult, 5)
        tp = round(close + atr_val * tp_mult, 5) if direction == "BUY" else round(close - atr_val * tp_mult, 5)
        expiry_note = ""

    minutes = timeframe_to_minutes(timeframe)
    entry_time = datetime.utcnow() + timedelta(minutes=minutes)

    if abs(score) >= 80:
        stars = 5
    elif abs(score) >= 60:
        stars = 4
    elif abs(score) >= 40:
        stars = 3
    elif abs(score) >= 20:
        stars = 2
    else:
        stars = 1

    return {
        "symbol": symbol,
        "direction": direction,
        "confluence_score": score,
        "stars": stars,
        "rsi": round(rsi_val, 2),
        "atr": round(atr_val, 5),
        "entry_price": close,
        "stop_loss": sl,
        "take_profit": tp,
        "expiry_note": expiry_note,
        "entry_time": entry_time.strftime("%H:%M:%S"),
        "timeframe": timeframe,
        "platform": platform,
        "details": details,
        "signal_time": datetime.utcnow().isoformat()
    }

@app.get("/market-prices")
def market_prices():
    cryptos = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
    crypto_prices = []
    for sym in cryptos:
        try:
            r = requests.get(f"https://api.binance.com/api/v3/ticker/24hr?symbol={sym}", timeout=10)
            data = r.json()
            crypto_prices.append({
                "symbol": sym,
                "price": float(data["lastPrice"]),
                "change": float(data["priceChangePercent"]),
                "volume": float(data["quoteVolume"])
            })
        except:
            pass
    forex_prices = [
        {"symbol": "EURUSD", "price": 1.0823, "change": 0.12,  "volume": 0},
        {"symbol": "GBPUSD", "price": 1.2634, "change": -0.05, "volume": 0},
        {"symbol": "USDJPY", "price": 144.15, "change": 0.23,  "volume": 0},
        {"symbol": "XAUUSD", "price": 2321.50,"change": 0.45,  "volume": 0}
    ]
    return {"crypto": crypto_prices, "forex": forex_prices}

@app.get("/")
def root():
    return {"status": "online"}
