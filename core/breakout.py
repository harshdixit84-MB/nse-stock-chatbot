"""
Volume Breakout strategy -- detection logic.

Catches momentum stocks breaking to a new N-day high on above-average
volume, without requiring a pullback first (which EMA Pullback would
require). Exit: stop-loss, target, or MAX_HOLD_DAYS (enforced later by
the backtest/live engine, not here).
"""

import pandas as pd

from config import (
    BREAKOUT_LOOKBACK_DAYS,
    BREAKOUT_VOLUME_MULT,
    BREAKOUT_MIN_AVG_VOLUME,
    RISK_REWARD_MULT,
    ATR_PERIOD,
    ATR_STOP_MULT,
)
from risk import compute_atr, atr_stop
from regime import is_bullish_on


def evaluate(df: pd.DataFrame, nifty_df: pd.DataFrame = None):
    """
    df must have columns: Open, High, Low, Close, Volume (most recent
    row last) and at least ~BREAKOUT_LOOKBACK_DAYS + 20 rows.

    Returns a dict if the LAST row closes above the prior
    BREAKOUT_LOOKBACK_DAYS high on above-average volume, otherwise None.
    """
    if df is None or len(df) < BREAKOUT_LOOKBACK_DAYS + 21:
        return None

    df = df.copy()
    df["AvgVol20"] = df["Volume"].rolling(20).mean()
    # Prior N-day high, EXCLUDING today, so today's close is measured
    # against the range that came before it.
    df["PriorHigh"] = df["High"].shift(1).rolling(BREAKOUT_LOOKBACK_DAYS).max()
    df["RecentLow"] = df["Low"].rolling(10).min()
    df["ATR"] = compute_atr(df, ATR_PERIOD)

    last = df.iloc[-1]
    close = last["Close"]
    prior_high = last["PriorHigh"]
    if pd.isna(prior_high):
        return None

    # 1. Must actually break out
    if close <= prior_high:
        return None

    # 2. Volume confirmation
    avg_vol20 = last["AvgVol20"]
    if pd.isna(avg_vol20) or avg_vol20 < BREAKOUT_MIN_AVG_VOLUME:
        return None
    if last["Volume"] < avg_vol20 * BREAKOUT_VOLUME_MULT:
        return None

    # 3. Market regime filter -- breakouts fail far more often when the
    # broader market is choppy/range-bound; skip rather than chase a
    # breakout against a downtrending index.
    if not is_bullish_on(nifty_df, last["Date"]):
        return None

    # ---- Setup confirmed: build the trade plan ----
    recent_low = last["RecentLow"]
    structural_stop = min(recent_low, prior_high)
    stop_loss = atr_stop(structural_stop, last["ATR"], ATR_STOP_MULT)
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
        "pattern": f"{BREAKOUT_LOOKBACK_DAYS}-Day Breakout",
        "breakout_level": round(float(prior_high), 2),
        "avg_volume_20d": int(avg_vol20),
        "nifty_regime": "nifty_uptrend" if nifty_df is not None else "not_checked",
    }
