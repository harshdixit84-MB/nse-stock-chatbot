"""
20/50 EMA Crossover strategy -- detection logic.

Distinct from EMA Pullback: this fires on a TREND CHANGE (20 EMA crossing
above 50 EMA), not a dip within an already-established uptrend. Exit is
whichever comes first: opposite crossover, stop-loss, target, or
MAX_HOLD_DAYS (enforced later by the backtest/live engine, not here).
"""

import pandas as pd

from config import (
    EMA_FAST,
    EMA_SLOW,
    CROSSOVER_LOOKBACK_DAYS,
    CROSSOVER_MIN_AVG_VOLUME,
    RISK_REWARD_MULT,
    STOP_BUFFER_PCT,
)


def evaluate(df: pd.DataFrame):
    """
    df must have columns: Open, High, Low, Close, Volume (most recent
    row last) and at least ~EMA_SLOW + CROSSOVER_LOOKBACK_DAYS rows.

    Returns a dict if a bullish 20/50 EMA crossover happened on the LAST
    row, otherwise None.
    """
    if df is None or len(df) < EMA_SLOW + CROSSOVER_LOOKBACK_DAYS + 1:
        return None

    df = df.copy()
    df["EMA20"] = df["Close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=EMA_SLOW, adjust=False).mean()
    df["AvgVol20"] = df["Volume"].rolling(20).mean()
    df["SwingLow"] = df["Low"].rolling(CROSSOVER_LOOKBACK_DAYS).min()

    last = df.iloc[-1]
    prev = df.iloc[-2]

    close = last["Close"]
    ema20, ema50 = last["EMA20"], last["EMA50"]
    prev_ema20, prev_ema50 = prev["EMA20"], prev["EMA50"]

    # 1. The crossover must happen ON this bar
    just_crossed_up = prev_ema20 <= prev_ema50 and ema20 > ema50
    if not just_crossed_up:
        return None

    # 2. Price should confirm the new trend
    if not (close > ema20 and close > ema50):
        return None

    # 3. Liquidity filter
    avg_vol20 = last["AvgVol20"]
    if pd.isna(avg_vol20) or avg_vol20 < CROSSOVER_MIN_AVG_VOLUME:
        return None

    # ---- Setup confirmed: build the trade plan ----
    swing_low = last["SwingLow"]
    stop_loss = min(swing_low, ema50) * (1 - STOP_BUFFER_PCT / 100)
    risk_per_share = close - stop_loss
    if risk_per_share <= 0:
        return None

    target = close + risk_per_share * RISK_REWARD_MULT

    return {
        "entry_price": round(float(close), 2),
        "stop_loss": round(float(stop_loss), 2),
        "target": round(float(target), 2),
        "risk_per_share": round(float(risk_per_share), 2),
        "reward_risk_ratio": round(float((target - close) / risk_per_share), 2),
        "pattern": "20/50 EMA Bullish Crossover",
        "ema20": round(float(ema20), 2),
        "ema50": round(float(ema50), 2),
        "avg_volume_20d": int(avg_vol20),
    }
