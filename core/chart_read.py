"""
core/chart_read.py

Always-on price-action chart read: trend direction, key support/
resistance levels, today's candlestick pattern, and a reversal watch --
independent of whether any of the 5 mechanical strategies have an active
setup today.

This exists specifically because "No active setup today" was a dead end.
A trader looking at a chart wants to know the underlying picture --
uptrend/downtrend/sideways, is a reversal brewing, where's the nearest
level -- whether or not a specific trade trigger fired. Reuses the same
swing-point/S-R/candlestick building blocks as price_action.py, so this
reads the chart the same way that strategy does, just without requiring
all of its conditions to line up before saying anything.
"""
import pandas as pd

from price_action import (
    _find_swing_points,
    _find_support_resistance,
    _is_hammer,
    _is_bullish_engulfing,
    _is_bullish_pin_bar,
)
from rsi_divergence import _compute_rsi
from config import PA_SWING_LOOKBACK, PA_SR_LOOKBACK_DAYS, PA_SR_TOUCH_TOLERANCE_PCT, PA_SR_MIN_TOUCHES


def _is_shooting_star(row):
    body_high = max(row["Close"], row["Open"])
    body_low = min(row["Close"], row["Open"])
    body_size = body_high - body_low
    upper_wick = row["High"] - body_high
    lower_wick = body_low - row["Low"]
    if body_size <= 0:
        return False
    return upper_wick >= body_size * 2 and lower_wick <= body_size * 0.5


def _is_bearish_engulfing(prev_row, row):
    prior_bullish = prev_row["Close"] > prev_row["Open"]
    is_bearish = row["Close"] < row["Open"]
    engulfs = row["Open"] >= prev_row["Close"] and row["Close"] <= prev_row["Open"]
    return prior_bullish and is_bearish and engulfs


def _is_bearish_pin_bar(row):
    body_high = max(row["Close"], row["Open"])
    body_low = min(row["Close"], row["Open"])
    body_size = body_high - body_low
    total_range = row["High"] - row["Low"]
    if total_range <= 0:
        return False
    upper_wick = row["High"] - body_high
    return upper_wick >= total_range * 0.6 and body_size <= total_range * 0.35


def _classify_trend(df, lookback):
    "Uptrend / Downtrend / Sideways via swing-point structure -- same fractal method as price_action.py, but reports all three outcomes instead of just Uptrend-or-nothing."
    swing_high_idxs, swing_low_idxs = _find_swing_points(df, lookback)
    if len(swing_high_idxs) < 2 or len(swing_low_idxs) < 2:
        return "Sideways / not enough structure yet", None, None

    last_two_highs = [float(df["High"].iloc[i]) for i in swing_high_idxs[-2:]]
    last_two_lows = [float(df["Low"].iloc[i]) for i in swing_low_idxs[-2:]]

    higher_highs = last_two_highs[1] > last_two_highs[0]
    higher_lows = last_two_lows[1] > last_two_lows[0]
    lower_highs = last_two_highs[1] < last_two_highs[0]
    lower_lows = last_two_lows[1] < last_two_lows[0]

    if higher_highs and higher_lows:
        trend = "Uptrend"
    elif lower_highs and lower_lows:
        trend = "Downtrend"
    else:
        trend = "Sideways / mixed structure"

    return trend, last_two_highs, last_two_lows


def _candle_pattern(df):
    "Checks the LAST candle for a recognizable reversal pattern, bullish or bearish -- whichever fires first."
    last = df.iloc[-1]
    prev = df.iloc[-2]

    if _is_hammer(last):
        return "Hammer", "bullish"
    if _is_bullish_engulfing(prev, last):
        return "Bullish Engulfing", "bullish"
    if _is_bullish_pin_bar(last):
        return "Bullish Pin Bar", "bullish"
    if _is_shooting_star(last):
        return "Shooting Star", "bearish"
    if _is_bearish_engulfing(prev, last):
        return "Bearish Engulfing", "bearish"
    if _is_bearish_pin_bar(last):
        return "Bearish Pin Bar", "bearish"
    return "No distinct reversal candle", "neutral"


def _rsi_zone(rsi_value):
    if rsi_value is None or pd.isna(rsi_value):
        return "n/a"
    if rsi_value >= 70:
        return "Overbought"
    if rsi_value <= 30:
        return "Oversold"
    return "Neutral"


def get_chart_read(df: pd.DataFrame) -> dict:
    """
    Always returns a full price-action read -- trend, key levels, candle
    pattern, RSI zone, and a reversal watch -- regardless of whether any
    of the 5 mechanical strategies have an active setup today.
    """
    min_rows = PA_SR_LOOKBACK_DAYS + PA_SWING_LOOKBACK * 2 + 5
    if df is None or len(df) < min_rows:
        return {"error": "Not enough history for a full chart read."}

    df = df.copy()
    close = float(df["Close"].iloc[-1])

    trend, last_two_highs, last_two_lows = _classify_trend(df, PA_SWING_LOOKBACK)

    support_levels, resistance_levels = _find_support_resistance(
        df, PA_SR_LOOKBACK_DAYS, PA_SR_TOUCH_TOLERANCE_PCT, PA_SR_MIN_TOUCHES
    )
    supports_below = [s for s in support_levels if s <= close]
    resistances_above = [r for r in resistance_levels if r > close]
    nearest_support = max(supports_below) if supports_below else None
    nearest_resistance = min(resistances_above) if resistances_above else None
    dist_to_support_pct = round((close - nearest_support) / nearest_support * 100, 2) if nearest_support else None
    dist_to_resistance_pct = round((nearest_resistance - close) / close * 100, 2) if nearest_resistance else None

    pattern_name, pattern_bias = _candle_pattern(df)

    rsi = _compute_rsi(df["Close"])
    rsi_now = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None
    rsi_zone = _rsi_zone(rsi_now)

    avg_vol20 = df["Volume"].rolling(20).mean().iloc[-1]
    vol_now = df["Volume"].iloc[-1]
    volume_vs_avg = "Above average" if (not pd.isna(avg_vol20) and vol_now > avg_vol20) else "Below/at average"

    # ---- Reversal watch: does today's candle push against the prevailing trend? ----
    reversal_watch = "None -- today's candle doesn't contradict the trend."
    if trend == "Uptrend" and pattern_bias == "bearish":
        near_res = dist_to_resistance_pct is not None and dist_to_resistance_pct <= 3
        reversal_watch = (
            f"Possible BEARISH reversal warning -- {pattern_name} printed while in an uptrend"
            + (f", right near resistance ({round(nearest_resistance, 2)})" if near_res else "")
            + ". Not confirmed until the next 1-2 candles follow through lower."
        )
    elif trend == "Downtrend" and pattern_bias == "bullish":
        near_sup = dist_to_support_pct is not None and dist_to_support_pct <= 3
        reversal_watch = (
            f"Possible BULLISH reversal warning -- {pattern_name} printed while in a downtrend"
            + (f", right near support ({round(nearest_support, 2)})" if near_sup else "")
            + ". Not confirmed until the next 1-2 candles follow through higher."
        )
    elif rsi_zone == "Overbought" and trend == "Uptrend":
        reversal_watch = "Uptrend intact, but RSI is Overbought -- momentum may be getting stretched, watch for a stall."
    elif rsi_zone == "Oversold" and trend == "Downtrend":
        reversal_watch = "Downtrend intact, but RSI is Oversold -- watch for a bounce/reversal candle."

    return {
        "trend": trend,
        "last_two_swing_highs": [round(h, 2) for h in last_two_highs] if last_two_highs else None,
        "last_two_swing_lows": [round(l, 2) for l in last_two_lows] if last_two_lows else None,
        "nearest_support": round(nearest_support, 2) if nearest_support else None,
        "distance_to_support_pct": dist_to_support_pct,
        "nearest_resistance": round(nearest_resistance, 2) if nearest_resistance else None,
        "distance_to_resistance_pct": dist_to_resistance_pct,
        "todays_candle_pattern": pattern_name,
        "todays_candle_bias": pattern_bias,
        "rsi_14": round(rsi_now, 2) if rsi_now is not None else None,
        "rsi_zone": rsi_zone,
        "volume_vs_20d_avg": volume_vs_avg,
        "reversal_watch": reversal_watch,
    }
