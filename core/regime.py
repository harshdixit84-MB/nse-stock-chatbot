"""
core/regime.py

Market-regime filter for EMA Crossover and Breakout.

Both strategies are, per the research review, explicitly regime-
dependent: trending markets favor them, range-bound/choppy markets
produce a disproportionate share of whipsaws (crossover) and false
breakouts (breakout). EMA Pullback and Price Action already require
their OWN stock to show trend structure, so they're left alone;
RSI Divergence is a reversal signal by nature and regime-gating it
would work against its purpose, so it's also left alone.

This module answers one question: was NIFTY 50 itself in an uptrend
(close > its own 200-EMA -- the standard index-trend filter) as of a
given date.
"""
import pandas as pd

NIFTY_TREND_EMA = 200


def is_bullish_on(nifty_df: pd.DataFrame, as_of_date) -> bool:
    """
    True if NIFTY 50 closed above its own EMA(200) as of the most
    recent NIFTY row on or before `as_of_date`.

    Used identically by live analysis (as_of_date = today) and the
    backtest's day-by-day walk (as_of_date = that historical day) --
    since this only ever looks at NIFTY rows <= as_of_date, the
    backtest gets no lookahead from this filter.

    Fails OPEN (returns True, i.e. "don't block the signal") if NIFTY
    data is missing or too short for a 200-EMA -- a NIFTY data outage
    should degrade to "regime filter not applied" rather than silently
    suppressing every crossover/breakout signal.
    """
    if nifty_df is None or nifty_df.empty:
        return True

    as_of_date = pd.Timestamp(as_of_date)
    eligible = nifty_df[nifty_df["Date"] <= as_of_date]
    if len(eligible) < NIFTY_TREND_EMA:
        return True

    ema200 = eligible["Close"].ewm(span=NIFTY_TREND_EMA, adjust=False).mean()
    return bool(eligible["Close"].iloc[-1] > ema200.iloc[-1])


def regime_label(nifty_df: pd.DataFrame) -> str:
    "Human-readable status for including in a signal's output dict."
    if nifty_df is None or nifty_df.empty:
        return "not_available"
    return "nifty_uptrend" if is_bullish_on(nifty_df, nifty_df["Date"].iloc[-1]) else "nifty_downtrend"
