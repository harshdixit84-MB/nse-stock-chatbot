"""
RSI Divergence strategy -- detection logic.

Adapted from Harsh/rsi_divergence.py (a port of TradingView's official RSI
Divergence indicator: RSI period 14, pivot lookback 5 bars each side, pivots
compared 5-60 bars apart, "Regular" divergence only). Only BULLISH
divergence is used here as a BUY entry signal, keeping this strategy's role
consistent with the other 3 (all long-entry signals).

Unlike the other 3 strategies, a divergence pivot can only be CONFIRMED
PIVOT_RIGHT bars after it forms (same lag TradingView's indicator has in
real time). evaluate() only fires if that confirmation lands exactly on
the LAST row -- a yes/no check on "did the signal just confirm on this
bar," not a lookback scan.
"""

import pandas as pd

from config import (
    RISK_REWARD_MULT,
    STOP_BUFFER_PCT,
    MIN_AVG_VOLUME,
)

RSI_PERIOD = 14
PIVOT_LEFT = 5
PIVOT_RIGHT = 5
RANGE_LOWER = 5
RANGE_UPPER = 60


def _compute_rsi(close, period=RSI_PERIOD):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _find_rsi_pivot_lows(rsi, left=PIVOT_LEFT, right=PIVOT_RIGHT):
    vals = rsi.values
    n = len(vals)
    lows = []

    for i in range(left, n - right):
        if pd.isna(vals[i]):
            continue
        window = vals[i - left:i + right + 1]
        if any(pd.isna(w) for w in window):
            continue
        if vals[i] == min(window):
            lows.append(i)

    return lows


def evaluate(df: pd.DataFrame):
    """
    df must have columns: Open, High, Low, Close, Volume (most recent
    row last) and enough history to cover RSI_PERIOD + RANGE_UPPER +
    PIVOT_LEFT + PIVOT_RIGHT bars.

    Returns a dict if a bullish RSI divergence's pivot gets CONFIRMED on
    the LAST row (i.e. the pivot formed PIVOT_RIGHT bars ago), otherwise
    None.
    """
    min_len = RSI_PERIOD + RANGE_UPPER + PIVOT_LEFT + PIVOT_RIGHT
    if df is None or len(df) < min_len:
        return None

    df = df.copy()
    rsi = _compute_rsi(df["Close"])
    lows = _find_rsi_pivot_lows(rsi)

    if len(lows) < 2:
        return None

    i1, i2 = lows[-2], lows[-1]

    # The pivot at i2 must be confirmed exactly on the LAST row.
    last_index = len(df) - 1
    if i2 + PIVOT_RIGHT != last_index:
        return None

    if not (RANGE_LOWER <= (i2 - i1) <= RANGE_UPPER):
        return None

    rsi1, rsi2 = rsi.iloc[i1], rsi.iloc[i2]
    price1, price2 = df["Low"].iloc[i1], df["Low"].iloc[i2]

    # Regular bullish divergence: price makes a LOWER low, RSI makes a HIGHER low
    if not (price2 < price1 and rsi2 > rsi1):
        return None

    # Liquidity filter
    avg_vol20 = df["Volume"].rolling(20).mean().iloc[-1]
    if pd.isna(avg_vol20) or avg_vol20 < MIN_AVG_VOLUME:
        return None

    last = df.iloc[-1]
    close = last["Close"]
    pivot_low_price = price2
    stop_loss = pivot_low_price * (1 - STOP_BUFFER_PCT / 100)
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
        "pattern": "Bullish RSI Divergence",
        "divergence_pivot_low": round(float(pivot_low_price), 2),
        "rsi_at_pivot": round(float(rsi2), 2),
        "avg_volume_20d": int(avg_vol20),
    }
