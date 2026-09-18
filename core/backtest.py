"""
core/backtest.py

Backtests all 5 strategies against a symbol's own history, so the
verdict for a stock is based on how each strategy has actually performed
on THAT stock, not a generic average across all stocks.

Methodology:
  - Pull ~5 years of daily data (within Angel SmartAPI's ~2000-day
    per-request cap, so no chaining needed).
  - Walk through history day-by-day. At each day, run each strategy's
    evaluate() using ONLY the data available up to that day (no
    lookahead).
  - When a strategy fires, simulate forward day-by-day (using the real
    future High/Low/Close) until stop-loss or target is hit, or
    MAX_HOLD_DAYS elapses -- whichever comes first. If a single day's
    range hits BOTH stop and target, the stop is assumed to hit first
    (the standard conservative backtesting convention).
  - Signals too close to the end of available history to be fully
    simulated are excluded from the stats entirely, rather than guessed at.

Required environment variables (same as analyze.py):
  ANGEL_API_KEY, ANGEL_CLIENT_ID, ANGEL_PASSWORD, ANGEL_TOTP_SECRET
"""
import time
import math
from datetime import datetime, timedelta

import pandas as pd

from analyze import _login, _get_token, _call_smartapi, _looks_like_auth_error, _refresh_or_relogin, _fetch_index_ohlcv
from config import MAX_HOLD_DAYS

import strategy
import ema_crossover
import breakout
import rsi_divergence
import price_action

BACKTEST_DAYS_BACK = 1825  # ~5 years -- stays within Angel SmartAPI's ~2000-day per-request cap

STRATEGIES = {
    "EMA Pullback": strategy,
    "EMA Crossover": ema_crossover,
    "Volume Breakout": breakout,
    "RSI Divergence": rsi_divergence,
    "Price Action": price_action,
}


def _fetch_extended_ohlcv(symbol: str, days_back: int = BACKTEST_DAYS_BACK):
    smart_api = _login()
    token = _get_token(symbol)
    if not token:
        return None

    to_date = datetime.now().strftime("%Y-%m-%d %H:%M")
    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d %H:%M")
    params = {
        "exchange": "NSE",
        "symboltoken": token,
        "interval": "ONE_DAY",
        "fromdate": from_date,
        "todate": to_date,
    }
    resp = _call_smartapi(smart_api.getCandleData, params)
    time.sleep(0.35)

    if _looks_like_auth_error(resp):
        smart_api = _refresh_or_relogin()
        resp = _call_smartapi(smart_api.getCandleData, params)
        time.sleep(0.35)

    if not resp or not resp.get("status") or not resp.get("data"):
        return None

    df = pd.DataFrame(resp["data"], columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])


def _simulate_trade(df: pd.DataFrame, entry_index: int, signal: dict):
    entry_price = signal["entry_price"]
    stop = signal["stop_loss"]
    target = signal["target"]
    risk = signal["risk_per_share"]

    for offset in range(1, MAX_HOLD_DAYS + 1):
        idx = entry_index + offset
        day = df.iloc[idx]
        hit_stop = day["Low"] <= stop
        hit_target = day["High"] >= target

        if hit_stop:
            exit_price = stop  # conservative: assume stop hit first if both hit same day
        elif hit_target:
            exit_price = target
        else:
            continue

        r_multiple = (exit_price - entry_price) / risk
        return {"r_multiple": r_multiple, "exit_index": idx}

    # Neither hit within MAX_HOLD_DAYS -- exit at close on the final day
    idx = entry_index + MAX_HOLD_DAYS
    exit_price = df.iloc[idx]["Close"]
    r_multiple = (exit_price - entry_price) / risk
    return {"r_multiple": r_multiple, "exit_index": idx}


def _run_one_strategy(df: pd.DataFrame, strategy_module, nifty_df: pd.DataFrame = None):
    results = []
    last_valid_i = len(df) - MAX_HOLD_DAYS - 1
    if last_valid_i < 1:
        return results

    for i in range(0, last_valid_i + 1):
        sub_df = df.iloc[: i + 1]
        # nifty_df is passed in FULL (not sliced to i) -- regime.is_bullish_on()
        # internally restricts to NIFTY rows <= this bar's own date, so this
        # introduces no lookahead: at step i, only NIFTY data up to that
        # historical day is ever looked at, exactly mirroring live analysis.
        signal = strategy_module.evaluate(sub_df, nifty_df=nifty_df)
        if signal is None:
            continue
        outcome = _simulate_trade(df, i, signal)
        results.append({
            "entry_date": str(df["Date"].iloc[i].date()),
            "entry_price": signal["entry_price"],
            "exit_date": str(df["Date"].iloc[outcome["exit_index"]].date()),
            "r_multiple": round(outcome["r_multiple"], 2),
        })

    return results


def _wilson_lower_bound(wins: int, n: int, z: float = 1.96) -> float:
    """
    Wilson score interval lower bound on a win rate -- the standard fix
    for comparing proportions estimated from different sample sizes.
    A strategy that's 3-for-3 (100%) and one that's 26-for-40 (65%)
    are NOT equally trustworthy; raw win rate treats them as if they
    are. This discounts a small sample toward 50% until it has enough
    signals to actually back up the raw number, so ranking by this
    value (instead of raw win_rate_pct) stops a handful of lucky
    trades from outranking a strategy with a large, solid track record.
    Returns a fraction in [0, 1] -- multiply by 100 for a percentage.
    """
    if n == 0:
        return 0.0
    p_hat = wins / n
    denom = 1 + z ** 2 / n
    center = p_hat + z ** 2 / (2 * n)
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z ** 2 / (4 * n)) / n)
    return max(0.0, (center - margin) / denom)


def _summarize(results):
    if not results:
        return {
            "signals": 0,
            "win_rate_pct": None,
            "win_rate_lcb_pct": None,
            "avg_r_multiple": None,
            "profit_factor": None,
        }

    wins = [r for r in results if r["r_multiple"] > 0]
    losses = [r for r in results if r["r_multiple"] <= 0]

    win_rate = len(wins) / len(results) * 100
    win_rate_lcb = _wilson_lower_bound(len(wins), len(results)) * 100
    avg_r = sum(r["r_multiple"] for r in results) / len(results)

    gross_win = sum(r["r_multiple"] for r in wins)
    gross_loss = abs(sum(r["r_multiple"] for r in losses))
    profit_factor = round(gross_win / gross_loss, 2) if gross_loss > 0 else None

    return {
        "signals": len(results),
        "win_rate_pct": round(win_rate, 1),
        "win_rate_lcb_pct": round(win_rate_lcb, 1),
        "avg_r_multiple": round(avg_r, 2),
        "profit_factor": profit_factor,
        "low_sample_warning": len(results) < 10,
    }


def backtest_symbol(symbol: str) -> dict:
    symbol = symbol.strip().upper()
    df = _fetch_extended_ohlcv(symbol)
    if df is None or df.empty:
        return {"symbol": symbol, "error": "Could not fetch history for this symbol."}

    output = {"symbol": symbol, "history_days": len(df), "strategies": {}}

    try:
        nifty_df = _fetch_index_ohlcv()
    except Exception:
        nifty_df = None
    output["nifty_regime_filter_available"] = nifty_df is not None

    for name, module in STRATEGIES.items():
        results = _run_one_strategy(df, module, nifty_df=nifty_df)
        output["strategies"][name] = _summarize(results)

    return output


if __name__ == "__main__":
    import json
    import sys
    sym = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE"
    print(json.dumps(backtest_symbol(sym), indent=2, default=str))
