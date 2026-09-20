"""
core/outlook.py

Top-down outlook for a single NSE stock:

    Nifty 50 trend  ->  the stock's sector-index trend  ->  the stock's own
    trend, patterns and indicators  ->  one combined bias plus a short
    plain-English summary of how the stock may move over the next ~5-10
    trading days.

How this differs from narrative.py (the "AI Narrative" button): that one
only looks at the stock's own chart. This one adds the two layers above
it (market + sector), scores all three, and gives an expected range and
an invalidation level. It is fully rule-based -- no LLM call, no Gemini
key, no extra timeout risk.

This is a probabilistic bias, not a prediction. Every result carries
an invalidation level for exactly that reason.

Data comes from the same Angel SmartAPI session analyze.py already uses:
  - stock candles      -> analyze._fetch_ohlcv (400 days)
  - Nifty 50 candles   -> analyze._fetch_index_ohlcv (cached in KV)
  - sector index candles -> _fetch_sector_index below (cached in KV, 6h)
Anything that can't be fetched degrades gracefully: the result says which
layer was skipped (see "data_notes") and the score re-weights over the
layers that are available. Nothing here ever raises out of get_outlook().

Required environment variables: same as analyze.py.
"""
import csv
import io
import json
import math
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

from chart_read import get_chart_read
from rsi_divergence import _compute_rsi

# ---------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------
MIN_ROWS_TREND = 60          # bars needed before a trend score is meaningful
RS_WINDOW_DAYS = 20          # relative-strength lookback
RANGE_HORIZONS = (5, 10)     # trading-day horizons for the expected range
SECTOR_OHLCV_TTL = 6 * 3600  # sector candles change once per session
MAP_TTL = 7 * 24 * 3600      # industry map / index tokens change rarely
SECTOR_HISTORY_DAYS = 400

WEIGHTS_FULL = {"market": 0.25, "sector": 0.25, "stock": 0.50}

INDUSTRY_MAP_KV_KEY = "industry_map_v1"
INDEX_TOKEN_KV_KEY = "index_token_v1"
EQ_TOKEN_KV_KEY = "nse_eq_tokens_v1"          # symbol -> Angel token for every NSE "-EQ" stock
INDEX_NAME_KV_KEY = "nse_index_names_v1"      # normalised index name -> Angel index token
OUTLOOK_KV_PREFIX = "outlook_v1_"
OUTLOOK_TTL = 3 * 3600                        # per-symbol result cache, shared across devices
SECTOR_OHLCV_KV_PREFIX = "sector_ohlcv_v1_"

# Nifty 500 constituent list -- its "Industry" column is what maps a
# stock to a sector index. Two mirrors; first one that answers wins.
INDUSTRY_CSV_URLS = [
    "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
    "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
]
CSV_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; nse-stock-chatbot/1.0)"}

# NSE "Industry" text (lower-cased substring) -> (label shown to the
# person, candidate Angel index names in preference order). Index names
# are compared after upper-casing and stripping non-alphanumerics, so
# "Nifty Fin Service" and "NIFTY FIN SERVICE" both become NIFTYFINSERVICE.
# Several spellings are listed because Angel's instrument master is not
# consistent about abbreviations.
SECTOR_RULES = [
    ("information technology", "IT", ["NIFTYIT"]),
    ("financial services", "Financial Services", ["NIFTYFINSERVICE", "NIFTYFINANCIALSERVICES", "NIFTYBANK"]),
    ("healthcare", "Healthcare / Pharma", ["NIFTYHEALTHCARE", "NIFTYPHARMA"]),
    ("automobile", "Auto", ["NIFTYAUTO"]),
    ("fast moving consumer", "FMCG", ["NIFTYFMCG"]),
    ("metals", "Metals & Mining", ["NIFTYMETAL"]),
    ("oil gas", "Oil, Gas & Energy", ["NIFTYENERGY", "NIFTYOILANDGAS", "NIFTYOILGAS"]),
    ("power", "Power / Energy", ["NIFTYENERGY"]),
    ("realty", "Realty", ["NIFTYREALTY"]),
    ("consumer durables", "Consumer Durables", ["NIFTYCONSRDURBL", "NIFTYCONSUMERDURABLES"]),
    ("media", "Media", ["NIFTYMEDIA"]),
    ("chemicals", "Chemicals", ["NIFTYCHEMICALS"]),
    ("construction", "Infrastructure", ["NIFTYINFRA", "NIFTYINFRASTRUCTURE"]),
]

_industry_map_cache = {"data": None}
_master_maps_cache = {"eq": None, "index": None}


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------
def _day_series(df):
    "Date column as tz-naive midnight timestamps, so stock/index/sector frames join cleanly."
    dates = pd.to_datetime(df["Date"])
    try:
        dates = dates.dt.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    return dates.dt.normalize()


def _ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def _macd_parts(close, fast=12, slow=26, signal=9):
    macd_line = _ema(close, fast) - _ema(close, slow)
    signal_line = _ema(macd_line, signal)
    return macd_line, signal_line, macd_line - signal_line


def _adx_atr(df, period=14):
    "Wilder ADX and ATR. Returns (adx_series, atr_series)."
    high, low, close = df["High"], df["Low"], df["Close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
    true_range = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
    ).max(axis=1)
    alpha = 1 / period
    atr = true_range.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    return adx, atr


def _f(value, digits=2):
    "float -> rounded float, or None for NaN/None."
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return round(float(value), digits)


def _clip(value, low=-100.0, high=100.0):
    return max(low, min(high, value))


def _jsonable(obj):
    "Round-trips through JSON so numpy scalars can't break the KV cache or the HTTP response."
    def default(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return str(o)
    return json.loads(json.dumps(obj, default=default))


def _bias_label(score):
    if score >= 55:
        return "Bullish"
    if score >= 20:
        return "Mildly bullish"
    if score > -20:
        return "Neutral / range-bound"
    if score > -55:
        return "Mildly bearish"
    return "Bearish"


# ---------------------------------------------------------------------
# Layer scoring: one trend read used for Nifty, the sector index, and
# the stock alike, so all three are judged on the same yardstick.
# ---------------------------------------------------------------------
def trend_read(df):
    """
    Scores a price series from -100 (strong downtrend) to +100 (strong
    uptrend) and labels it. Components (points, +/-):
      close vs EMA50 (20) | close vs EMA200 (20, only if 200+ bars) |
      EMA20 vs EMA50 (15) | EMA50 slope over 10 bars (15) |
      MACD line vs signal (10) | RSI(14) above 55 / below 45 (10) |
      20-bar price change beyond +/-2% (10)
    ADX(14) only describes how strong the trend is (low < 20 <= moderate < 25 <= strong); it doesn't move the score.
    """
    if df is None or len(df) < MIN_ROWS_TREND:
        return {"error": "Not enough history for a trend read."}

    close = df["Close"]
    last = float(close.iloc[-1])
    ema20, ema50 = _ema(close, 20), _ema(close, 50)
    ema200 = _ema(close, 200) if len(df) >= 200 else None
    macd_line, signal_line, _hist = _macd_parts(close)
    rsi = _compute_rsi(close)
    adx, _atr = _adx_atr(df)

    points, max_points = 0.0, 0.0

    def add(weight, condition_up, condition_down):
        nonlocal points, max_points
        max_points += weight
        if condition_up:
            points += weight
        elif condition_down:
            points -= weight

    add(20, last > ema50.iloc[-1], last < ema50.iloc[-1])
    if ema200 is not None:
        add(20, last > ema200.iloc[-1], last < ema200.iloc[-1])
    add(15, ema20.iloc[-1] > ema50.iloc[-1], ema20.iloc[-1] < ema50.iloc[-1])
    slope_ref = ema50.iloc[-11]
    add(15, ema50.iloc[-1] > slope_ref, ema50.iloc[-1] < slope_ref)
    add(10, macd_line.iloc[-1] > signal_line.iloc[-1], macd_line.iloc[-1] < signal_line.iloc[-1])
    rsi_now = rsi.iloc[-1]
    add(10, (not pd.isna(rsi_now)) and rsi_now >= 55, (not pd.isna(rsi_now)) and rsi_now <= 45)
    change_20 = (last / float(close.iloc[-21]) - 1) * 100
    add(10, change_20 > 2, change_20 < -2)

    score = round(100 * points / max_points) if max_points else 0
    if score >= 40:
        label = "Uptrend"
    elif score <= -40:
        label = "Downtrend"
    else:
        label = "Sideways / mixed"

    adx_now = adx.iloc[-1]
    if pd.isna(adx_now):
        strength = "unknown"
    elif adx_now >= 25:
        strength = "strong"
    elif adx_now >= 20:
        strength = "moderate"
    else:
        strength = "low"

    return {
        "label": label,
        "score": int(score),
        "strength": strength,
        "adx": _f(adx_now, 1),
        "rsi_14": _f(rsi_now, 1),
        "close": _f(last),
        "ema20": _f(ema20.iloc[-1]),
        "ema50": _f(ema50.iloc[-1]),
        "ema200": _f(ema200.iloc[-1]) if ema200 is not None else None,
        "change_20d_pct": _f(change_20, 1),
    }


def relative_strength(df_a, df_b, window=RS_WINDOW_DAYS):
    "How A has done vs B over `window` bars, from the A/B price ratio. None if the two can't be aligned."
    if df_a is None or df_b is None:
        return None
    a = pd.Series(df_a["Close"].values, index=_day_series(df_a))
    b = pd.Series(df_b["Close"].values, index=_day_series(df_b))
    a = a[~a.index.duplicated(keep="last")]
    b = b[~b.index.duplicated(keep="last")]
    joined = pd.concat([a, b], axis=1, join="inner").dropna()
    if len(joined) < window + 5:
        return None
    joined.columns = ["a", "b"]
    ratio = joined["a"] / joined["b"]
    change = (float(ratio.iloc[-1]) / float(ratio.iloc[-1 - window]) - 1) * 100
    above_avg = float(ratio.iloc[-1]) > float(_ema(ratio, window).iloc[-1])
    if change > 1 and above_avg:
        label = "Outperforming"
    elif change < -1 and not above_avg:
        label = "Lagging"
    else:
        label = "In line"
    return {"label": label, "change_pct": round(change, 2), "window_days": window}


# ---------------------------------------------------------------------
# Stock-only: patterns, levels, expected range
# ---------------------------------------------------------------------
def stock_patterns(df, chart=None):
    "List of {name, bias} for what the chart is doing right now. bias = bullish | bearish | neutral."
    found = []
    if df is None or len(df) < 60:
        return found

    close, high, low, volume = df["Close"], df["High"], df["Low"], df["Volume"]
    last = float(close.iloc[-1])
    prior_high = float(high.iloc[-21:-1].max())
    prior_low = float(low.iloc[-21:-1].min())
    avg_vol = float(volume.iloc[-21:-1].mean())
    vol_ratio = float(volume.iloc[-1]) / avg_vol if avg_vol else 0.0

    if last > prior_high:
        if vol_ratio >= 1.5:
            found.append({"name": f"Breakout above 20-day high on strong volume ({vol_ratio:.1f}x average)", "bias": "bullish"})
        else:
            found.append({"name": "Breakout above 20-day high, but volume is not confirming", "bias": "neutral"})
    elif last < prior_low:
        found.append({"name": "Breakdown below 20-day low", "bias": "bearish"})

    high_52w = float(high.iloc[-252:].max())
    low_52w = float(low.iloc[-252:].min())
    if last >= 0.95 * high_52w:
        found.append({"name": "Trading within 5% of its 52-week high", "bias": "bullish"})
    elif last <= 1.05 * low_52w:
        found.append({"name": "Trading within 5% of its 52-week low", "bias": "bearish"})

    ema20, ema50 = _ema(close, 20), _ema(close, 50)
    e20, e50 = float(ema20.iloc[-1]), float(ema50.iloc[-1])
    if e20 > e50 and last > e50:
        if abs(last - e50) / e50 <= 0.02:
            found.append({"name": "Pulling back to the 50 EMA inside an uptrend", "bias": "bullish"})
        elif abs(last - e20) / e20 <= 0.015:
            found.append({"name": "Pulling back to the 20 EMA inside an uptrend", "bias": "bullish"})

    diff = ema20 - ema50
    recent = diff.iloc[-6:]
    if (recent.iloc[0] <= 0) and (recent.iloc[-1] > 0):
        found.append({"name": "Fresh 20/50 EMA bullish crossover (last 5 days)", "bias": "bullish"})
    elif (recent.iloc[0] >= 0) and (recent.iloc[-1] < 0):
        found.append({"name": "Fresh 20/50 EMA bearish crossover (last 5 days)", "bias": "bearish"})

    _adx, atr = _adx_atr(df)
    if len(atr.dropna()) > 25:
        atr_now, atr_then = float(atr.iloc[-1]), float(atr.iloc[-21])
        range_10 = (float(high.iloc[-10:].max()) - float(low.iloc[-10:].min())) / last
        if atr_then and atr_now / atr_then < 0.8 and range_10 < 0.08:
            found.append({"name": "Volatility contracting into a tight range (coiling)", "bias": "neutral"})

    if chart and "error" not in chart:
        pattern = chart.get("todays_candle_pattern")
        bias = chart.get("todays_candle_bias")
        if bias in ("bullish", "bearish") and pattern:
            found.append({"name": f"Today's candle: {pattern}", "bias": bias})

    return found


def expected_range(df, horizons=RANGE_HORIZONS):
    """
    Typical (1-sigma, ~68% of the time) price range using the last 20
    days' daily volatility scaled by sqrt(days). Symmetric on purpose:
    volatility says how far it may move, not which way.
    """
    if df is None or len(df) < 25:
        return None
    returns = np.log(df["Close"].astype(float)).diff().dropna().iloc[-20:]
    sigma = float(returns.std())
    if not sigma or math.isnan(sigma):
        return None
    last = float(df["Close"].iloc[-1])
    out = {}
    for h in horizons:
        move = sigma * math.sqrt(h)
        out[f"{h}d"] = {
            "low": round(last * math.exp(-move), 2),
            "high": round(last * math.exp(move), 2),
            "move_pct": round(move * 100, 1),
        }
    return out


def _levels(df, chart):
    "Nearest support/resistance, from chart_read when available, else the 20-day low/high."
    support = resistance = None
    source = "20-day low/high"
    if chart and "error" not in chart:
        support = chart.get("nearest_support")
        resistance = chart.get("nearest_resistance")
        if support is not None or resistance is not None:
            source = "swing-based support/resistance"
    if support is None:
        support = _f(df["Low"].iloc[-21:-1].min())
    if resistance is None:
        resistance = _f(df["High"].iloc[-21:-1].max())
    return {"support": support, "resistance": resistance, "source": source}


# ---------------------------------------------------------------------
# Combine everything
# ---------------------------------------------------------------------
def _pattern_adjustment(patterns):
    bullish = sum(1 for p in patterns if p["bias"] == "bullish")
    bearish = sum(1 for p in patterns if p["bias"] == "bearish")
    return int(_clip((bullish - bearish) * 5, -15, 15))


def _alignment_note(market, sector, stock_label):
    "One line on whether the three layers agree -- the part a stock-only read can't give."
    if not market or "error" in market:
        return None
    m, s = market["label"], (sector or {}).get("label") if sector and "error" not in sector else None
    if stock_label == "Uptrend" and m == "Downtrend":
        return "The stock is strong against a weak market: a broad market fall can still drag it down."
    if stock_label == "Downtrend" and m == "Uptrend":
        return "The stock is weak even though the market is strong: that is relative weakness, not just market noise."
    if stock_label == "Uptrend" and m == "Uptrend" and s == "Uptrend":
        return "Market, sector and stock are all in uptrends: the layers agree."
    if stock_label == "Downtrend" and m == "Downtrend" and s == "Downtrend":
        return "Market, sector and stock are all in downtrends: the layers agree."
    if s == "Downtrend" and stock_label == "Uptrend":
        return "The stock is rising against a weak sector; sector weakness is a headwind."
    if s == "Uptrend" and stock_label == "Downtrend":
        return "The stock is falling while its sector is strong; that points to a stock-specific problem."
    return None


def _combine(layer_scores):
    "Weighted average over whichever layers exist, weights re-normalised."
    available = {k: v for k, v in layer_scores.items() if v is not None}
    total_weight = sum(WEIGHTS_FULL[k] for k in available)
    if not total_weight:
        return 0
    return sum(WEIGHTS_FULL[k] * v for k, v in available.items()) / total_weight


def decide_action(composite, market, stock, patterns, levels, last_close):
    """
    Turns the outlook into one call: BUY / HOLD / SELL, plus a plain reason.
    HOLD means "wait / no fresh entry" -- if you already own it, keep it.

    SELL  composite <= -40, or the stock is in a downtrend below its 50 EMA,
          or it just broke below its 20-day low while the composite is negative.
    BUY   composite >= 50 AND the stock is in an uptrend above its 50 EMA AND
          the market isn't in a downtrend AND RSI isn't stretched (<= 72) AND
          the invalidation level (nearest support) is within 8% of price,
          so the risk is a sensible size.
    HOLD  everything else, with the reason that stopped it being a BUY.

    These thresholds are judgment calls, not backtested numbers.
    """
    rsi = stock.get("rsi_14")
    ema50 = stock.get("ema50")
    label = stock.get("label")
    below_ema50 = ema50 is not None and last_close < ema50
    above_ema50 = ema50 is not None and last_close > ema50
    broke_down = any(p["name"].startswith("Breakdown below") for p in patterns)

    if composite <= -40:
        return {"label": "SELL", "reason": f"Overall score {composite:+d}: trend, sector and market lean down. Avoid new buying; exit if you hold."}
    if label == "Downtrend" and below_ema50:
        return {"label": "SELL", "reason": "The stock is in a downtrend and trading below its 50 EMA. Avoid new buying; exit if you hold."}
    if broke_down and composite < 0:
        return {"label": "SELL", "reason": "The stock just broke below its 20-day low while the overall score is negative."}

    support = levels.get("support")
    risk_pct = (last_close - support) / last_close * 100 if support else None
    market_down = bool(market) and "error" not in market and market.get("label") == "Downtrend"

    if composite >= 50 and label == "Uptrend" and above_ema50:
        if market_down:
            return {"label": "HOLD", "reason": "The stock looks strong, but Nifty is in a downtrend. Wait for the market to stabilise before buying."}
        if rsi is not None and rsi > 72:
            return {"label": "HOLD", "reason": f"Trend is up but RSI {rsi} is stretched. Wait for a pullback instead of chasing."}
        if risk_pct is not None and risk_pct > 8:
            return {"label": "HOLD", "reason": f"Trend is up, but the invalidation level is {risk_pct:.1f}% below price, too wide for a sensible stop. Wait for a pullback."}
        return {"label": "BUY", "reason": "Market, sector and stock trends line up, momentum is not stretched, and the stop level is close."}

    if composite >= 20:
        return {"label": "HOLD", "reason": "Leaning positive, but not strong or clean enough for a fresh buy yet. Wait for confirmation."}
    return {"label": "HOLD", "reason": "No clear edge either way right now. Wait."}


def _fmt(value):
    return f"₹{value:,.2f}" if value is not None else "n/a"


def _phrase(label):
    "Trend label -> a phrase that reads naturally after 'is in'."
    return {"Uptrend": "an uptrend", "Downtrend": "a downtrend"}.get(label, "a sideways / mixed phase")


def _lower_first(text):
    return text[:1].lower() + text[1:] if text else text


def _build_summary(symbol, action, bias, score, market, sector_name, sector, sector_rs, stock, patterns,
                   rng, levels, alignment, missing_layers):
    lines = []
    lines.append(f"{symbol}: {action['label']}. {action['reason']}")
    lines.append(f"Overall bias: {bias} for the next 5-10 trading days (score {score:+d} on a -100 to +100 scale).")

    if market and "error" not in market:
        lines.append(f"Market: Nifty is in {_phrase(market['label'])} (trend strength {market['strength']}, RSI {market['rsi_14']}).")

    if sector_name and sector and "error" not in sector:
        rs_text = ""
        if sector_rs:
            rs_text = ", and is " + {
                "Outperforming": "outperforming Nifty",
                "Lagging": "lagging Nifty",
            }.get(sector_rs["label"], "moving in line with Nifty")
        lines.append(f"Sector: {sector_name} is in {_phrase(sector['label'])}{rs_text}.")

    stock_line = f"Stock: in {_phrase(stock['label'])} (trend strength {stock['strength']}), RSI {stock['rsi_14']}"
    if patterns:
        stock_line += "; " + "; ".join(_lower_first(p["name"]) for p in patterns[:3])
    lines.append(stock_line + ".")

    if alignment:
        lines.append(alignment)

    if rng:
        r5, r10 = rng.get("5d"), rng.get("10d")
        parts = []
        if r5:
            parts.append(f"5 days {_fmt(r5['low'])} to {_fmt(r5['high'])}")
        if r10:
            parts.append(f"10 days {_fmt(r10['low'])} to {_fmt(r10['high'])}")
        lines.append("Typical range (about 2 in 3 times): " + "; ".join(parts) + ".")

    support, resistance = levels["support"], levels["resistance"]
    if bias in ("Bullish", "Mildly bullish"):
        lines.append(f"Resistance {_fmt(resistance)}; the bullish view weakens on a close below {_fmt(support)}.")
    elif bias in ("Bearish", "Mildly bearish"):
        lines.append(f"Support {_fmt(support)}; the bearish view weakens on a close above {_fmt(resistance)}.")
    else:
        lines.append(f"Likely to stay between support {_fmt(support)} and resistance {_fmt(resistance)} until one of them breaks.")

    if missing_layers:
        lines.append("Note: " + " and ".join(missing_layers) + " could not be included, so the score rests on the remaining layers.")

    return "\n".join(lines)


def build_outlook(symbol, df, nifty_df=None, sector_name=None, sector_df=None, data_notes=None):
    """
    Pure function (no network): everything get_outlook() does once the
    candles are in hand. Kept separate so it can be tested offline.
    """
    data_notes = list(data_notes or [])
    stock = trend_read(df)
    if "error" in stock:
        return {"symbol": symbol, "error": stock["error"]}

    chart = None
    try:
        chart = get_chart_read(df)
    except Exception:
        chart = None

    market = trend_read(nifty_df) if nifty_df is not None else None
    sector = trend_read(sector_df) if sector_df is not None else None

    patterns = stock_patterns(df, chart)
    stock_rs_nifty = relative_strength(df, nifty_df) if nifty_df is not None else None
    stock_rs_sector = relative_strength(df, sector_df) if sector_df is not None else None
    sector_rs_nifty = relative_strength(sector_df, nifty_df) if (sector_df is not None and nifty_df is not None) else None

    rs_adj = 0
    if stock_rs_nifty:
        rs_adj += {"Outperforming": 5, "Lagging": -5}.get(stock_rs_nifty["label"], 0)
    if stock_rs_sector:
        rs_adj += {"Outperforming": 5, "Lagging": -5}.get(stock_rs_sector["label"], 0)
    stock_score = int(_clip(stock["score"] + _pattern_adjustment(patterns) + rs_adj))

    market_score = market["score"] if market and "error" not in market else None
    sector_score = sector["score"] if sector and "error" not in sector else None

    missing = []
    if market_score is None:
        missing.append("the Nifty trend")
    if sector_score is None:
        missing.append("the sector trend")

    composite = int(round(_clip(_combine({"market": market_score, "sector": sector_score, "stock": stock_score}))))
    bias = _bias_label(composite)
    levels = _levels(df, chart)
    rng = expected_range(df)
    alignment = _alignment_note(market, sector, stock["label"])
    last_close = float(df["Close"].iloc[-1])
    action = decide_action(composite, market, stock, patterns, levels, last_close)

    summary = _build_summary(
        symbol, action, bias, composite, market, sector_name, sector, sector_rs_nifty,
        stock, patterns, rng, levels, alignment, missing,
    )

    result = {
        "symbol": symbol,
        "as_of": str(pd.to_datetime(df["Date"].iloc[-1]).date()),
        "last_close": _f(df["Close"].iloc[-1]),
        "action": action,
        "bias": bias,
        "score": composite,
        "summary": summary,
        "market": market,
        "sector": ({"name": sector_name, **sector, "vs_nifty": sector_rs_nifty} if sector and "error" not in sector else None),
        "stock": {
            **stock,
            "score_with_patterns": stock_score,
            "vs_nifty": stock_rs_nifty,
            "vs_sector": stock_rs_sector,
        },
        "patterns": patterns,
        "levels": levels,
        "expected_range": rng,
        "alignment": alignment,
        "data_notes": data_notes,
        "disclaimer": "A rule-based read of price data, not a prediction or advice. The Buy/Hold/Sell thresholds are judgment calls and have not been backtested. The cut-off level matters more than the label.",
    }
    return _jsonable(result)


# ---------------------------------------------------------------------
# Data fetching (network). Everything below fails soft.
# ---------------------------------------------------------------------
def _normalise_name(text):
    return "".join(ch for ch in str(text).upper() if ch.isalnum())


def _rows_to_df(rows):
    df = pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])


def _df_to_cache(df):
    out = df.copy()
    out["Date"] = out["Date"].astype(str)
    return out.to_dict(orient="records")


def _cache_to_df(records):
    df = pd.DataFrame(records)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def _load_industry_map():
    "symbol -> NSE industry, from the Nifty 500 list. KV-cached for a week. {} if unreachable."
    import analyze

    if _industry_map_cache["data"] is not None:
        return _industry_map_cache["data"]

    cached = analyze._kv_get(INDUSTRY_MAP_KV_KEY)
    if cached:
        _industry_map_cache["data"] = cached
        return cached

    for url in INDUSTRY_CSV_URLS:
        try:
            resp = requests.get(url, headers=CSV_HEADERS, timeout=8)
            resp.raise_for_status()
            mapping = {}
            for row in csv.DictReader(io.StringIO(resp.text)):
                symbol = (row.get("Symbol") or "").strip().upper()
                industry = (row.get("Industry") or "").strip()
                if symbol and industry:
                    mapping[symbol] = industry
            if len(mapping) > 100:
                analyze._kv_set(INDUSTRY_MAP_KV_KEY, mapping, MAP_TTL)
                _industry_map_cache["data"] = mapping
                return mapping
        except Exception:
            continue
    return {}


def _sector_for(symbol):
    "-> (label, [index name aliases], industry) or (None, None, industry_or_None)."
    industry = _load_industry_map().get(symbol)
    if not industry:
        return None, None, None
    lowered = industry.lower()
    for needle, label, aliases in SECTOR_RULES:
        if needle in lowered:
            return label, aliases, industry
    return None, None, industry


def _load_token_maps():
    """
    (eq_map, index_by_name) built from Angel's instrument master -- a
    ~30 MB download that analyze._get_token repeats on every cold start.
    Here it is downloaded at most once per day: both small maps are kept
    in memory and in KV, so every later request (and every other symbol)
    skips the download. Returns (None, None) if the master can't be read.
    """
    import analyze

    if _master_maps_cache["eq"] is not None:
        return _master_maps_cache["eq"], _master_maps_cache["index"]

    eq_map = analyze._kv_get(EQ_TOKEN_KV_KEY)
    index_map = analyze._kv_get(INDEX_NAME_KV_KEY)
    if eq_map and index_map:
        _master_maps_cache["eq"], _master_maps_cache["index"] = eq_map, index_map
        return eq_map, index_map

    try:
        resp = requests.get(analyze.INSTRUMENT_MASTER_URL, timeout=60)
        resp.raise_for_status()
        instruments = resp.json()
    except Exception:
        return None, None

    eq_map, index_map = {}, {}
    for inst in instruments:
        if inst.get("exch_seg") != "NSE":
            continue
        token = str(inst.get("token", ""))
        symbol = inst.get("symbol", "")
        if symbol.endswith("-EQ"):
            eq_map.setdefault(symbol[:-3], token)
        elif inst.get("instrumenttype") in ("AMXIDX", "", None) and token.startswith("999"):
            for field in ("name", "symbol"):
                key = _normalise_name(inst.get(field, ""))
                if key:
                    index_map.setdefault(key, token)

    if eq_map:
        _master_maps_cache["eq"], _master_maps_cache["index"] = eq_map, index_map
        analyze._kv_set(EQ_TOKEN_KV_KEY, eq_map, 24 * 3600)
        analyze._kv_set(INDEX_NAME_KV_KEY, index_map, MAP_TTL)
        return eq_map, index_map
    return None, None


def _prime_stock_token(symbol):
    """
    Puts the stock's Angel token into analyze's own token cache so
    analyze._fetch_ohlcv doesn't download the whole master itself.
    Returns False only when the master was read and the symbol isn't in
    it (i.e. definitely not an NSE equity); True otherwise.
    """
    import analyze

    if symbol in analyze._token_cache:
        return True
    eq_map, _ = _load_token_maps()
    if eq_map is None:
        return True  # couldn't read the master; let analyze try its own way
    token = eq_map.get(symbol)
    if token:
        analyze._token_cache[symbol] = token
        return True
    return False


def _resolve_index_token(aliases):
    """
    Finds a working Angel token for the first alias that returns candles.
    Returns (token, candles_df) or (None, None). The working token is
    remembered in KV, so this normally costs one candle call.
    """
    import analyze

    token_map = analyze._kv_get(INDEX_TOKEN_KV_KEY) or {}
    for alias in aliases:
        if alias in token_map:
            df = _fetch_index_candles(token_map[alias])
            if df is not None:
                return token_map[alias], df

    _eq, index_by_name = _load_token_maps()
    if not index_by_name:
        return None, None

    for alias in aliases:
        amx_token = index_by_name.get(alias)
        if not amx_token:
            continue
        # Angel's own docs disagree on which token form serves candles
        # (see NIFTY_TOKEN_CANDIDATES in analyze.py) -- try both.
        for candidate in (amx_token, "26" + amx_token[-3:]):
            df = _fetch_index_candles(candidate)
            if df is not None:
                token_map[alias] = candidate
                analyze._kv_set(INDEX_TOKEN_KV_KEY, token_map, MAP_TTL)
                return candidate, df
    return None, None


def _fetch_index_candles(token, days_back=SECTOR_HISTORY_DAYS):
    import analyze

    try:
        smart_api = analyze._login()
        params = {
            "exchange": "NSE",
            "symboltoken": token,
            "interval": "ONE_DAY",
            "fromdate": (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d %H:%M"),
            "todate": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        resp = analyze._call_smartapi(smart_api.getCandleData, params)
        time.sleep(0.35)
        if analyze._looks_like_auth_error(resp):
            smart_api = analyze._refresh_or_relogin()
            resp = analyze._call_smartapi(smart_api.getCandleData, params)
            time.sleep(0.35)
        if resp and resp.get("status") and resp.get("data"):
            df = _rows_to_df(resp["data"])
            return df if not df.empty else None
    except Exception:
        return None
    return None


def _fetch_sector_index(aliases):
    "Sector index candles, KV-cached for 6 hours. None if anything fails."
    import analyze

    cache_key = SECTOR_OHLCV_KV_PREFIX + aliases[0]
    cached = analyze._kv_get(cache_key)
    if cached:
        try:
            df = _cache_to_df(cached)
            if not df.empty:
                return df
        except Exception:
            pass

    try:
        _token, df = _resolve_index_token(aliases)
    except Exception:
        return None
    if df is not None:
        analyze._kv_set(cache_key, _df_to_cache(df), SECTOR_OHLCV_TTL)
    return df


def _compute_outlook(symbol):
    import analyze

    notes = []

    if not _prime_stock_token(symbol):
        return {"symbol": symbol, "error": "Could not fetch data. Check this is a valid NSE equity symbol."}

    try:
        df = analyze._fetch_ohlcv(symbol)
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}
    if df is None or df.empty:
        return {"symbol": symbol, "error": "Could not fetch data. Check this is a valid NSE equity symbol."}

    try:
        nifty_df = analyze._fetch_index_ohlcv()
    except Exception:
        nifty_df = None
    if nifty_df is None:
        notes.append("Nifty data could not be fetched right now.")

    sector_name, sector_df = None, None
    try:
        label, aliases, industry = _sector_for(symbol)
        if label:
            sector_df = _fetch_sector_index(aliases)
            if sector_df is not None:
                sector_name = label
            else:
                notes.append(f"Sector index for {label} could not be fetched.")
        elif industry:
            notes.append(f"No matching NSE sector index for industry '{industry}'.")
        else:
            notes.append("This stock is not in the Nifty 500 list, so its sector could not be identified.")
    except Exception:
        notes.append("Sector lookup failed.")

    try:
        return build_outlook(symbol, df, nifty_df, sector_name, sector_df, notes)
    except Exception as e:
        return {"symbol": symbol, "error": f"Outlook calculation failed: {e}"}


def get_outlook(symbol, use_cache=True):
    """
    Fetch stock + Nifty + sector candles and build the outlook. Never raises.
    Successful results are cached in KV for OUTLOOK_TTL so the dashboard's
    Buy/Hold/Sell badges (one call per ticker) don't hit Angel again for
    a symbol someone already loaded a few hours ago. Errors are not cached.
    """
    import analyze

    symbol = symbol.strip().upper()
    key = OUTLOOK_KV_PREFIX + symbol

    if use_cache:
        cached = analyze._kv_get(key)
        if cached and cached.get("action") and "error" not in cached:
            cached["cached"] = True
            return cached

    result = _compute_outlook(symbol)
    if "error" not in result:
        analyze._kv_set(key, result, OUTLOOK_TTL)
    return result


if __name__ == "__main__":
    import sys

    sym = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE"
    print(json.dumps(get_outlook(sym), indent=2, default=str))
