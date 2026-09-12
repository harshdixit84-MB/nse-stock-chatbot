"""
Price Action strategy -- detection logic.

Deliberately does NOT use EMA, RSI, or MACD in the trigger itself --
those are already covered by the other 4 strategies. This is a pure
price-action read, the way a discretionary trader would look at a bare
candlestick chart:

  1. Trend STRUCTURE via swing highs/lows (higher highs + higher lows),
     not a moving average -- the textbook price-action definition of an
     uptrend.
  2. A horizontal support level: a price zone the stock has actually
     tested 2+ times before, not just wherever an indicator happens to
     sit today.
  3. A bullish reversal candle (Hammer / Bullish Engulfing / Bullish Pin
     Bar) printing AT that support zone -- the "someone defended this
     level today" signal.
  4. Volume confirmation + the same liquidity floor as the other
     strategies.

All four have to line up on the same day for a signal to fire.
"""
import pandas as pd

from config import (
    PA_SWING_LOOKBACK,
    PA_SR_LOOKBACK_DAYS,
    PA_SR_TOUCH_TOLERANCE_PCT,
    PA_SR_MIN_TOUCHES,
    PA_NEAR_SUPPORT_PCT,
    PA_MIN_AVG_VOLUME,
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


def _is_bullish_pin_bar(row):
    # Looser than a hammer -- just needs a long lower rejection wick and a
    # small body, no cap on the upper wick.
    body_high = max(row["Close"], row["Open"])
    body_low = min(row["Close"], row["Open"])
    body_size = body_high - body_low
    total_range = row["High"] - row["Low"]
    if total_range <= 0:
        return False
    lower_wick = body_low - row["Low"]
    return lower_wick >= total_range * 0.6 and body_size <= total_range * 0.35


def _find_swing_points(df, lookback):
    "Fractal swing highs/lows: a bar higher/lower than `lookback` bars on each side."
    highs, lows = df["High"].values, df["Low"].values
    n = len(df)
    swing_high_idxs, swing_low_idxs = [], []
    for i in range(lookback, n - lookback):
        window_h = highs[i - lookback: i + lookback + 1]
        window_l = lows[i - lookback: i + lookback + 1]
        if highs[i] == window_h.max():
            swing_high_idxs.append(i)
        if lows[i] == window_l.min():
            swing_low_idxs.append(i)
    return swing_high_idxs, swing_low_idxs


def _trend_structure(df, lookback):
    "Higher highs + higher lows over the last two swing points each -> Uptrend."
    swing_high_idxs, swing_low_idxs = _find_swing_points(df, lookback)
    if len(swing_high_idxs) < 2 or len(swing_low_idxs) < 2:
        return False, None, None

    last_two_highs = [float(df["High"].iloc[i]) for i in swing_high_idxs[-2:]]
    last_two_lows = [float(df["Low"].iloc[i]) for i in swing_low_idxs[-2:]]

    higher_highs = last_two_highs[1] > last_two_highs[0]
    higher_lows = last_two_lows[1] > last_two_lows[0]

    return (higher_highs and higher_lows), last_two_highs, last_two_lows


def _find_support_resistance(df, lookback_days, tolerance_pct, min_touches):
    "Cluster recent swing lows/highs into levels tested `min_touches`+ times."
    recent = df.tail(lookback_days)

    def cluster(values):
        values = sorted(values)
        clusters = []
        for v in values:
            placed = False
            for c in clusters:
                if abs(v - c["level"]) / c["level"] * 100 <= tolerance_pct:
                    c["touches"] += 1
                    c["level"] = (c["level"] * (c["touches"] - 1) + v) / c["touches"]
                    placed = True
                    break
            if not placed:
                clusters.append({"level": v, "touches": 1})
        return sorted(c["level"] for c in clusters if c["touches"] >= min_touches)

    return cluster(recent["Low"].tolist()), cluster(recent["High"].tolist())


def evaluate(df: pd.DataFrame):
    """
    df must have columns: Open, High, Low, Close, Volume (most recent
    row last) and at least ~PA_SR_LOOKBACK_DAYS rows.

    Returns a dict describing the setup if the LAST row qualifies,
    otherwise returns None.
    """
    min_rows = PA_SR_LOOKBACK_DAYS + PA_SWING_LOOKBACK * 2 + 5
    if df is None or len(df) < min_rows:
        return None

    df = df.copy()
    df["AvgVol20"] = df["Volume"].rolling(20).mean()

    last = df.iloc[-1]
    prev = df.iloc[-2]
    close = last["Close"]

    # 1. Trend structure -- no EMA/indicator involved
    is_uptrend, last_two_highs, last_two_lows = _trend_structure(df, PA_SWING_LOOKBACK)
    if not is_uptrend:
        return None

    # 2. A horizontal support level the stock has actually tested before
    support_levels, resistance_levels = _find_support_resistance(
        df, PA_SR_LOOKBACK_DAYS, PA_SR_TOUCH_TOLERANCE_PCT, PA_SR_MIN_TOUCHES
    )
    supports_below = [s for s in support_levels if s <= close]
    if not supports_below:
        return None
    nearest_support = max(supports_below)
    dist_to_support_pct = (close - nearest_support) / nearest_support * 100
    if dist_to_support_pct > PA_NEAR_SUPPORT_PCT:
        return None

    # 3. A bullish reversal candle printing AT that level
    is_hammer = _is_hammer(last)
    is_engulfing = _is_bullish_engulfing(prev, last)
    is_pin_bar = _is_bullish_pin_bar(last)
    if not (is_hammer or is_engulfing or is_pin_bar):
        return None

    # 4. Liquidity + volume confirmation
    avg_vol20 = last["AvgVol20"]
    if pd.isna(avg_vol20) or avg_vol20 < PA_MIN_AVG_VOLUME:
        return None

    # ---- All filters passed: build the trade plan ----
    stop_loss = nearest_support * (1 - STOP_BUFFER_PCT / 100)
    risk_per_share = close - stop_loss
    if risk_per_share <= 0:
        return None

    resistances_above = [r for r in resistance_levels if r > close]
    next_resistance = min(resistances_above) if resistances_above else None
    target_2r = close + risk_per_share * RISK_REWARD_MULT
    target = max(target_2r, next_resistance) if next_resistance else target_2r

    pattern = "Hammer" if is_hammer else ("Bullish Engulfing" if is_engulfing else "Bullish Pin Bar")

    return {
        "entry_price": round(float(close), 2),
        "stop_loss": round(float(stop_loss), 2),
        "target": round(float(target), 2),
        "risk_per_share": round(float(risk_per_share), 2),
        "reward_risk_ratio": round(float((target - close) / risk_per_share), 2),
        "pattern": f"{pattern} at Support",
        "support_level": round(float(nearest_support), 2),
        "next_resistance": round(float(next_resistance), 2) if next_resistance else None,
        "trend_structure": "Higher Highs + Higher Lows",
        "last_two_swing_highs": [round(h, 2) for h in last_two_highs],
        "last_two_swing_lows": [round(l, 2) for l in last_two_lows],
        "avg_volume_20d": int(avg_vol20),
    }
