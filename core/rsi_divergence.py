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

Two extra preconditions beyond the raw RSI/price divergence (added after
a research review flagged that divergence alone, with no trend context
or confirmation, is one of the weaker signals in isolation):
  1. A real prior decline (>= MIN_PRIOR_DECLINE_PCT from the recent high)
     into the first pivot -- divergence is meant to flag weakening
     momentum inside an actual downtrend, not fire on two arbitrary
     RSI wiggles in a flat/choppy stretch.
  2. A bullish reversal candle (hammer or engulfing) AT the second pivot
     itself -- confirms actual buying showed up at the low, not just
     that RSI curled up.
"""

import pandas as pd

from config import (
    RISK_REWARD_MULT,
    MIN_AVG_VOLUME,
    ATR_PERIOD,
    ATR_STOP_MULT,
)
from risk import compute_atr, atr_stop
from strategy import _is_hammer, _is_bullish_engulfing

# ---- RSI Divergence config ----
RSI_PERIOD = 14
PIVOT_LEFT = 5
PIVOT_RIGHT = 5
RANGE_LOWER = 5
RANGE_UPPER = 60

# A divergence only means something if it follows a genuine downtrend --
# research on this strategy is consistent that a raw divergence signal
# taken in isolation, with no established prior trend, is unreliable;
# it's meant to flag WEAKENING of an existing decline, not fire as a
# standalone reversal trigger. Require the stock to have actually fallen
# at least this % from its recent high into the divergence's first pivot
# before treating the divergence as valid.
MIN_PRIOR_DECLINE_PCT = 8.0


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


def evaluate(df: pd.DataFrame, nifty_df: pd.DataFrame = None):
    """
    df must have columns: Open, High, Low, Close, Volume (most recent
    row last) and enough history to cover RSI_PERIOD + RANGE_UPPER +
    PIVOT_LEFT + PIVOT_RIGHT bars.

    nifty_df is accepted for a uniform call signature across all 5
    strategies but unused here -- this is a reversal signal by nature,
    so gating it on the broader market's trend would work against its
    purpose.

    Returns a dict if a bullish RSI divergence's pivot gets CONFIRMED on
    the LAST row (i.e. the pivot formed PIVOT_RIGHT bars ago), otherwise
    None.
    """
    min_len = RSI_PERIOD + RANGE_UPPER + PIVOT_LEFT + PIVOT_RIGHT
    if df is None or len(df) < min_len:
        return None

    df = df.copy()
    rsi = _compute_rsi(df["Close"])
    atr = compute_atr(df, ATR_PERIOD)
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

    # Prior-downtrend precondition -- a divergence only means something as
    # a warning of WEAKENING momentum inside an actual decline. Require a
    # real prior fall from the recent high into the first pivot, not just
    # two arbitrary RSI wiggles.
    lookback_start = max(0, i1 - RANGE_UPPER)
    swing_high_before = df["High"].iloc[lookback_start:i1 + 1].max()
    if swing_high_before <= 0:
        return None
    decline_pct = (swing_high_before - price1) / swing_high_before * 100
    if decline_pct < MIN_PRIOR_DECLINE_PCT:
        return None

    # Candlestick confirmation at the second (lower) pivot itself -- research
    # on this strategy is clear that a raw divergence alone is much weaker
    # than one confirmed by actual reversal price action at the low.
    pivot_row = df.iloc[i2]
    pivot_prev_row = df.iloc[i2 - 1]
    is_hammer = _is_hammer(pivot_row)
    is_engulfing = _is_bullish_engulfing(pivot_prev_row, pivot_row)
    if not (is_hammer or is_engulfing):
        return None

    # Liquidity filter
    avg_vol20 = df["Volume"].rolling(20).mean().iloc[-1]
    if pd.isna(avg_vol20) or avg_vol20 < MIN_AVG_VOLUME:
        return None

    last = df.iloc[-1]
    close = last["Close"]
    pivot_low_price = price2
    stop_loss = atr_stop(pivot_low_price, atr.iloc[-1], ATR_STOP_MULT)
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
        "pattern": "Bullish RSI Divergence" + (" + Hammer" if is_hammer else " + Bullish Engulfing"),
        "divergence_pivot_low": round(float(pivot_low_price), 2),
        "rsi_at_pivot": round(float(rsi2), 2),
        "prior_decline_pct": round(float(decline_pct), 1),
        "avg_volume_20d": int(avg_vol20),
    }
