"""
core/backtest.py

Backtests all 4 strategies against a symbol's own history, so the
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
from datetime import datetime, timedelta

import pandas as pd

from analyze import _login, _get_token
from config import MAX_HOLD_DAYS

import strategy
import ema_crossover
import breakout
import rsi_divergence

BACKTEST_DAYS_BACK = 1825  # ~5 years -- stays within Angel SmartAPI's ~2000-day per-request cap

STRATEGIES = {
    "EMA Pullback": strategy,
    "EMA Crossover": ema_crossover,
    "Volume Breakout": breakout,
    "RSI Divergence": rsi_divergence,
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
    resp = smart_api.getCandleData(params)
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


def _run_one_strategy(df: pd.DataFrame, strategy_module):
    results = []
    last_valid_i = len(df) - MAX_HOLD_DAYS - 1
    if last_valid_i < 1:
        return results

    for i in range(0, last_valid_i + 1):
        sub_df = df.iloc[: i + 1]
        signal = strategy_module.evaluate(sub_df)
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


def _summarize(results):
    if not results:
        return {"signals": 0, "win_rate_pct": None, "avg_r_multiple": None, "profit_factor": None}

    wins = [r for r in results if r["r_multiple"] > 0]
    losses = [r for r in results if r["r_multiple"] <= 0]

    win_rate = len(wins) / len(results) * 100
    avg_r = sum(r["r_multiple"] for r in results) / len(results)

    gross_win = sum(r["r_multiple"] for r in wins)
    gross_loss = abs(sum(r["r_multiple"] for r in losses))
    profit_factor = round(gross_win / gross_loss, 2) if gross_loss > 0 else None

    return {
        "signals": len(results),
        "win_rate_pct": round(win_rate, 1),
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

    for name, module in STRATEGIES.items():
        results = _run_one_strategy(df, module)
        output["strategies"][name] = _summarize(results)

    return output


if __name__ == "__main__":
    import json
    import sys
    sym = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE"
    print(json.dumps(backtest_symbol(sym), indent=2, default=str))
