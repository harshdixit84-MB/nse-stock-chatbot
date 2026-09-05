"""
core/verdict.py

Combines a symbol's live signal check with its own historical backtest
into a single call: among whichever strategies are actively firing
TODAY, report the one with the highest backtested win rate for THIS
stock -- with full trade values -- plus a MACD momentum check as a
supporting (not deciding) confirmation.

Reuses a SINGLE ~5-year data fetch for everything (today's check, the
historical backtest, and MACD), rather than fetching 3 times separately.
"""
import pandas as pd

from backtest import _fetch_extended_ohlcv, _run_one_strategy, _summarize, STRATEGIES


def _compute_macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _macd_confirmation(df: pd.DataFrame) -> dict:
    macd_line, signal_line, histogram = _compute_macd(df["Close"])

    macd_now, signal_now = macd_line.iloc[-1], signal_line.iloc[-1]
    hist_now, hist_prev = histogram.iloc[-1], histogram.iloc[-2]

    bullish_direction = macd_now > signal_now
    rising_momentum = hist_now > hist_prev

    if bullish_direction and rising_momentum:
        status = "Confirmed"
    elif bullish_direction or rising_momentum:
        status = "Partially confirmed"
    else:
        status = "Not confirmed"

    return {
        "status": status,
        "macd_line": round(float(macd_now), 3),
        "signal_line": round(float(signal_now), 3),
        "histogram": round(float(hist_now), 3),
        "histogram_rising": bool(rising_momentum),
    }


def get_verdict(symbol: str) -> dict:
    symbol = symbol.strip().upper()
    df = _fetch_extended_ohlcv(symbol)
    if df is None or df.empty:
        return {"symbol": symbol, "error": "Could not fetch data. Check this is a valid NSE equity symbol."}

    last = df.iloc[-1]
    live_setups = {}
    ranked = []

    for name, module in STRATEGIES.items():
        today_setup = module.evaluate(df)
        live_setups[name] = today_setup

        history_results = _run_one_strategy(df, module)
        stats = _summarize(history_results)

        ranked.append({
            "strategy": name,
            "active_today": today_setup is not None,
            "signals": stats.get("signals", 0),
            "win_rate_pct": stats.get("win_rate_pct"),
            "avg_r_multiple": stats.get("avg_r_multiple"),
            "profit_factor": stats.get("profit_factor"),
            "low_sample_warning": stats.get("low_sample_warning", True),
        })

    ranked_sorted = sorted(
        ranked,
        key=lambda r: (r["win_rate_pct"] is None, -(r["win_rate_pct"] or 0)),
    )

    active_with_data = [r for r in ranked_sorted if r["active_today"] and r["win_rate_pct"] is not None]
    active_without_data = [r for r in ranked_sorted if r["active_today"] and r["win_rate_pct"] is None]

    result = {
        "symbol": symbol,
        "as_of": str(last["Date"].date()),
        "last_close": round(float(last["Close"]), 2),
        "strategy_ranking": ranked_sorted,
    }

    if active_with_data:
        pick = active_with_data[0]
        result["verdict"] = "BUY"
        result["strategy"] = pick["strategy"]
        result["trade_plan"] = live_setups[pick["strategy"]]
        result["backtest"] = {
            "win_rate_pct": pick["win_rate_pct"],
            "signals": pick["signals"],
            "avg_r_multiple": pick["avg_r_multiple"],
            "profit_factor": pick["profit_factor"],
            "low_sample_warning": pick["low_sample_warning"],
        }
    elif active_without_data:
        pick = active_without_data[0]
        result["verdict"] = "BUY_NO_TRACK_RECORD"
        result["strategy"] = pick["strategy"]
        result["trade_plan"] = live_setups[pick["strategy"]]
        result["note"] = "This setup is active today, but has no historical signals on this stock to back-test -- treat with extra caution."
    else:
        result["verdict"] = "NO_ACTIVE_SETUP"
        result["note"] = "None of the 4 strategies have a setup active on this stock today. See strategy_ranking for what's historically worked best, to know what to watch for."

    result["momentum_confirmation"] = _macd_confirmation(df)

    return result


if __name__ == "__main__":
    import json
    import sys
    sym = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE"
    print(json.dumps(get_verdict(sym), indent=2, default=str))
