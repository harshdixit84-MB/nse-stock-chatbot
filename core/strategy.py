"""
EMA Pullback swing strategy -- detection logic.

This mirrors the Pine Script version exactly, so backtests done on
TradingView and results produced here should agree.
"""

import pandas as pd

from config import (
    EMA_FAST,
    EMA_SLOW,
    SWING_LOOKBACK_DAYS,
    PULLBACK_MIN_PCT,
    PULLBACK_MAX_PCT,
    EMA_TOLERANCE_PCT,
    VOLUME_CONFIRM_MULT,
    MIN_AVG_VOLUME,
    RISK_REWARD_MULT,
    STOP_BUFFER_PCT,
)


def _is_hammer(row):
    body_high = max(row["Close"], row["Open"])
    body_low = min(row["Close"], row["Open"])
    body_size = body_high - body_low
    upper_wick = row["High"] - body_high
    lower_wick = body_low - row["Low"]
    if body_size <= 0:
        return False
    return lower_wick >= body_size * 2 and upper_wick <= body_size * 0.5


def _is_bullish_engulfing(prev_row, row):
    prior_bearish = prev_row["Close"] < prev_row["Open"]
    is_bullish = row["Close"] > row["Open"]
    engulfs = row["Close"] >= prev_row["Open"] and row["Open"] <= prev_row["Close"]
    return prior_bearish and is_bullish and engulfs


def evaluate(df: pd.DataFrame):
    """
    df must have columns: Open, High, Low, Close, Volume (most recent
    row last) and at least ~60 rows of history.

    Returns a dict describing the setup if the LAST row qualifies,
    otherwise returns None.
    """
    if df is None or len(df) < EMA_SLOW + SWING_LOOKBACK_DAYS:
        return None

    df = df.copy()
    df["EMA20"] = df["Close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=EMA_SLOW, adjust=False).mean()
    df["AvgVol20"] = df["Volume"].rolling(20).mean()
    df["SwingHigh"] = df["High"].rolling(SWING_LOOKBACK_DAYS).max()
    df["PullbackLow5"] = df["Low"].rolling(5).min()

    last = df.iloc[-1]
    prev = df.iloc[-2]

    close = last["Close"]
    ema20 = last["EMA20"]
    ema50 = last["EMA50"]

    # 1. Trend filter
    uptrend = close > ema20 and ema20 > ema50 and close > ema50
    if not uptrend:
        return None

    # 2. Pullback filter
    swing_high = last["SwingHigh"]
    if swing_high <= 0:
        return None
    pullback_pct = (swing_high - close) / swing_high * 100
    valid_pullback = PULLBACK_MIN_PCT <= pullback_pct <= PULLBACK_MAX_PCT
    if not valid_pullback:
        return None

    # 3. Price near EMA20 or EMA50
    dist_ema20_pct = abs(close - ema20) / ema20 * 100
    dist_ema50_pct = abs(close - ema50) / ema50 * 100
    near_ema = dist_ema20_pct <= EMA_TOLERANCE_PCT or dist_ema50_pct <= EMA_TOLERANCE_PCT
    if not near_ema:
        return None

    # 4. Liquidity filter
    avg_vol20 = last["AvgVol20"]
    if pd.isna(avg_vol20) or avg_vol20 < MIN_AVG_VOLUME:
        return None

    # 5. Bullish reversal candle
    is_hammer = _is_hammer(last)
    is_engulfing = _is_bullish_engulfing(prev, last)
    if not (is_hammer or is_engulfing):
        return None

    # 6. Volume confirmation
    if last["Volume"] < avg_vol20 * VOLUME_CONFIRM_MULT:
        return None

    # ---- All filters passed: build the trade plan ----
    pullback_low = last["PullbackLow5"]
    stop_loss = max(pullback_low, ema50) * (1 - STOP_BUFFER_PCT / 100)
    risk_per_share = close - stop_loss
    if risk_per_share <= 0:
        return None

    target_2r = close + risk_per_share * RISK_REWARD_MULT
    target = max(target_2r, swing_high)

    return {
        "entry_price": round(float(close), 2),
        "stop_loss": round(float(stop_loss), 2),
        "target": round(float(target), 2),
        "risk_per_share": round(float(risk_per_share), 2),
        "reward_risk_ratio": round(float((target - close) / risk_per_share), 2),
        "pullback_pct": round(float(pullback_pct), 2),
        "swing_high": round(float(swing_high), 2),
        "pattern": "Hammer" if is_hammer else "Bullish Engulfing",
        "ema20": round(float(ema20), 2),
        "ema50": round(float(ema50), 2),
        "avg_volume_20d": int(avg_vol20),
    }
