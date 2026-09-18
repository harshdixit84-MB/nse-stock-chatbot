"""
core/risk.py

Shared ATR-based stop-loss buffer, used by every strategy instead of a
fixed percentage buffer (STOP_BUFFER_PCT).

Why: a fixed percentage buffer is the same width whether the stock is
calm or violently volatile that week. Research on this is consistent --
a 1% (or 0.5%) buffer sized for a quiet stock gets hit constantly on a
volatile one, and a buffer wide enough to survive a volatile stretch is
needlessly loose once things calm down. Scaling the buffer to the
stock's own recent True Range (ATR) fixes both directions at once.

Each strategy keeps its own structural stop level (swing low, EMA50,
support level, etc.) -- this module only changes how far BELOW that
structural level the actual stop is placed.
"""
import pandas as pd


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    True Range = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
    ATR = Wilder's smoothed moving average of True Range (same
    smoothing convention as the RSI calc already used in
    rsi_divergence.py, for consistency).
    """
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def atr_stop(structural_level: float, atr_value: float, mult: float) -> float:
    """
    Place the stop `mult` * ATR below a structural level, instead of a
    fixed percentage of price below it.

    Falls back to the raw structural level (old behavior, no buffer)
    if ATR isn't available yet -- e.g. not enough history -- so callers
    never need their own None/NaN check.
    """
    if atr_value is None or pd.isna(atr_value):
        return structural_level
    return structural_level - (atr_value * mult)
