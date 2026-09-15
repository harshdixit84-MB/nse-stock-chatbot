"""
core/narrative.py

Adds a plain-English narrative layer ON TOP of the existing quantitative
verdict (verdict.py) -- explains the chart, does not compute or override
any trading numbers. The quant verdict (strategy, win rate, buy/target/
stop, MACD) stays the source of truth; the LLM is explicitly told not to
invent its own price levels, and is handed the SAME support/resistance/
trend/RSI numbers chart_read.py already computed, so it narrates from
real, already-verified numbers instead of guessing new ones.

Two data windows, per the design doc:
  - ~180 days (6 months), compressed to WEEKLY closes -- the broader
    trend and prior swing highs/lows.
  - ~30 days, full daily OHLCV -- the near-term pullback/consolidation/
    breakout structure most relevant to a 15-day-max swing hold.

Reuses the SAME df verdict.py already fetched via _fetch_extended_ohlcv
-- this module never fetches its own data, matching the project's
existing "single fetch, reuse everywhere" design.

Environment variables:
  GEMINI_API_KEY   -- required. No key, no narrative (fails soft, see
                      get_narrative below -- never breaks the quant
                      verdict response). Get one free, no credit card,
                      at https://aistudio.google.com/apikey
  NARRATIVE_MODEL  -- optional, defaults to gemini-3-flash-preview.
                      Kept as an env var, not hardcoded, since Google
                      rotates model names/aliases every few months --
                      Gemini 2.5 models (the prior default) are being
                      shut down Oct 2026, which is why this changed.
"""
import os

import pandas as pd
import requests

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
NARRATIVE_MODEL = os.environ.get("NARRATIVE_MODEL", "gemini-3-flash-preview")

WEEKLY_WINDOW_DAYS = 180
DAILY_WINDOW_DAYS = 30


def build_price_windows(df):
    """
    Returns (weekly_df, daily_df) built from the tail of the df verdict.py
    already fetched -- no separate fetch.

    _fetch_extended_ohlcv (Angel SmartAPI) returns "Date" as a plain
    column with a RangeIndex, not a DatetimeIndex -- unlike a yfinance-
    style frame. Resampling to weekly needs a real DatetimeIndex, so
    that's set here, on a copy, without touching what the caller has.
    """
    indexed = df.set_index(pd.to_datetime(df["Date"]))

    daily_df = indexed.tail(DAILY_WINDOW_DAYS)

    weekly_source = indexed.tail(WEEKLY_WINDOW_DAYS)
    weekly_df = weekly_source.resample("W").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum",
    }).dropna()

    return weekly_df, daily_df


def _format_ohlcv_table(frame, max_rows):
    "Compact CSV-style text block -- keeps the prompt small and cheap."
    rows = frame.tail(max_rows)
    lines = ["date,open,high,low,close,volume"]
    for idx, row in rows.iterrows():
        lines.append(f"{idx.date()},{row['Open']:.2f},{row['High']:.2f},{row['Low']:.2f},{row['Close']:.2f},{int(row['Volume'])}")
    return "\n".join(lines)


def build_narrative_prompt(symbol, verdict_result, weekly_df, daily_df):
    chart = verdict_result.get("chart_read", {}) or {}
    verdict_type = verdict_result.get("verdict")
    plan = verdict_result.get("trade_plan")
    strategy = verdict_result.get("strategy")
    bt = verdict_result.get("backtest")

    ground_truth_lines = [
        f"Symbol: {symbol}",
        f"As of: {verdict_result.get('as_of')}",
        f"Last close: {verdict_result.get('last_close')}",
        f"Quant verdict: {verdict_type}" + (f" via {strategy}" if strategy else ""),
    ]
    if plan:
        ground_truth_lines.append(
            f"Trade plan (DO NOT CHANGE THESE NUMBERS): entry {plan['entry_price']}, "
            f"stop {plan['stop_loss']}, target {plan['target']}, reward:risk 1:{plan['reward_risk_ratio']}"
        )
    if bt:
        ground_truth_lines.append(f"Backtested win rate for this strategy on this stock: {bt['win_rate_pct']}% over {bt['signals']} signals")
    if chart and "error" not in chart:
        ground_truth_lines.append(f"Trend (from swing structure): {chart.get('trend')}")
        if chart.get("nearest_support") is not None:
            ground_truth_lines.append(f"Nearest support: {chart['nearest_support']} ({chart.get('distance_to_support_pct')}% below)")
        if chart.get("nearest_resistance") is not None:
            ground_truth_lines.append(f"Nearest resistance: {chart['nearest_resistance']} ({chart.get('distance_to_resistance_pct')}% above)")
        ground_truth_lines.append(f"Today's candle: {chart.get('todays_candle_pattern')} ({chart.get('todays_candle_bias')})")
        ground_truth_lines.append(f"RSI(14): {chart.get('rsi_14')} -- {chart.get('rsi_zone')}")
        ground_truth_lines.append(f"Reversal watch: {chart.get('reversal_watch')}")

    ground_truth_block = "\n".join(ground_truth_lines)
    weekly_table = _format_ohlcv_table(weekly_df, max_rows=26)
    daily_table = _format_ohlcv_table(daily_df, max_rows=30)

    return f"""You are annotating a stock chart for an Indian swing trader (max 15-day hold).

GROUND TRUTH (already computed -- do NOT invent, recompute, or contradict any of these numbers or levels):
{ground_truth_block}

WEEKLY CLOSES (last ~6 months, for the broader trend):
{weekly_table}

DAILY OHLCV (last ~30 days, for near-term structure):
{daily_table}

Write a 150-200 word plain-English chart narrative covering:
1. The broader trend (from the weekly data) -- where is price relative to its recent swing highs/lows?
2. The recent daily structure -- pullback, consolidation, or breakout, and how it fits the broader trend.
3. Two or three conditional scenarios in "if X then Y" form (e.g. "if price holds above the support level noted above, the setup stays intact; if it breaks below, expect a retest of the next level down").
4. One closing line: does the recent price action support or weaken confidence in the quant verdict above?

Do not state any price level, support/resistance figure, or trade number that isn't already given above. If the ground truth doesn't support a confident read, say so plainly instead of guessing."""


def get_narrative(symbol, verdict_result, df):
    """
    Builds the two data windows from the SAME df verdict.py already
    fetched, builds the prompt, and calls the Gemini API (free tier,
    no credit card). Returns {"text": "...", "model": "..."} on success,
    or {"error": "..."} on any failure -- this NEVER raises, so a
    narrative failure can never break the underlying quant verdict
    response it's attached to.
    """
    if not GEMINI_API_KEY:
        return {"error": "GEMINI_API_KEY is not set -- narrative unavailable. Get a free key (no credit card) at https://aistudio.google.com/apikey"}

    try:
        weekly_df, daily_df = build_price_windows(df)
        if weekly_df.empty or daily_df.empty:
            return {"error": "Not enough price history for a narrative."}

        prompt = build_narrative_prompt(symbol, verdict_result, weekly_df, daily_df)

        resp = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{NARRATIVE_MODEL}:generateContent",
            headers={
                "x-goog-api-key": GEMINI_API_KEY,
                "Content-Type": "application/json",
            },
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "maxOutputTokens": 2048,
                    # Gemini 3 models think by default, and those thinking
                    # tokens are drawn from the SAME maxOutputTokens budget
                    # as the visible answer -- at "low" they still reason a
                    # little, but leave enough room for the actual ~150-200
                    # word narrative to finish instead of cutting off mid-
                    # sentence. This is just narrating pre-computed numbers,
                    # not solving anything that needs deep reasoning.
                    "thinkingConfig": {"thinkingLevel": "low"},
                },
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        candidates = data.get("candidates", [])
        if not candidates:
            block_reason = data.get("promptFeedback", {}).get("blockReason")
            if block_reason:
                return {"error": f"Gemini declined to respond (reason: {block_reason})."}
            return {"error": "Gemini returned no candidates."}

        parts = candidates[0].get("content", {}).get("parts", [])
        text = "\n".join(p.get("text", "") for p in parts).strip()
        if not text:
            return {"error": "Gemini returned an empty response."}
        return {"text": text, "model": NARRATIVE_MODEL}
    except Exception as e:
        return {"error": f"Narrative generation failed: {e}"}
