"""
core/analyze.py

Live, on-demand analysis of a single NSE stock symbol.
Fetches live daily OHLCV via Angel One SmartAPI (same source your
scanner uses) and runs it through each strategy module.

Required environment variables:
  ANGEL_API_KEY, ANGEL_CLIENT_ID, ANGEL_PASSWORD, ANGEL_TOTP_SECRET
"""
import os
import time
from datetime import datetime, timedelta

import pandas as pd
import pyotp
import requests
from SmartApi import SmartConnect

import strategy
   import ema_crossover
   import breakout
   import rsi_divergence

INSTRUMENT_MASTER_URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
HISTORY_DAYS_BACK = 400  # enough calendar days to cover EMA50 + lookback windows comfortably

_smart_api = None
_token_cache = {}


def _login():
    global _smart_api
    if _smart_api is not None:
        return _smart_api
    smart_api = SmartConnect(api_key=os.environ["ANGEL_API_KEY"])
    totp_code = pyotp.TOTP(os.environ["ANGEL_TOTP_SECRET"]).now()
    session = smart_api.generateSession(
        os.environ["ANGEL_CLIENT_ID"], os.environ["ANGEL_PASSWORD"], totp_code
    )
    if not session or not session.get("status"):
        raise RuntimeError(f"Angel One login failed: {session}")
    _smart_api = smart_api
    return smart_api


def _get_token(symbol: str):
    if symbol in _token_cache:
        return _token_cache[symbol]
    resp = requests.get(INSTRUMENT_MASTER_URL, timeout=60)
    resp.raise_for_status()
    for inst in resp.json():
        if inst.get("exch_seg") == "NSE" and inst.get("symbol", "") == f"{symbol}-EQ":
            _token_cache[symbol] = inst.get("token")
            return _token_cache[symbol]
    return None


def _fetch_ohlcv(symbol: str):
    smart_api = _login()
    token = _get_token(symbol)
    if not token:
        return None

    to_date = datetime.now().strftime("%Y-%m-%d %H:%M")
    from_date = (datetime.now() - timedelta(days=HISTORY_DAYS_BACK)).strftime("%Y-%m-%d %H:%M")
    params = {
        "exchange": "NSE",
        "symboltoken": token,
        "interval": "ONE_DAY",
        "fromdate": from_date,
        "todate": to_date,
    }
    resp = smart_api.getCandleData(params)
    time.sleep(0.35)  # respect SmartAPI's rate limit

    if not resp or not resp.get("status") or not resp.get("data"):
        return None

    df = pd.DataFrame(resp["data"], columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])


def analyze_symbol(symbol: str) -> dict:
    symbol = symbol.strip().upper()
    df = _fetch_ohlcv(symbol)
    if df is None or df.empty:
        return {"symbol": symbol, "error": "Could not fetch data. Check this is a valid NSE equity symbol."}

    last = df.iloc[-1]
    ema20 = df["Close"].ewm(span=20, adjust=False).mean().iloc[-1]
    ema50 = df["Close"].ewm(span=50, adjust=False).mean().iloc[-1]

    return {
        "symbol": symbol,
        "last_close": round(float(last["Close"]), 2),
        "as_of": str(last["Date"].date()),
        "ema20": round(float(ema20), 2),
        "ema50": round(float(ema50), 2),
        "pullback_setup": strategy.evaluate(df),
       "crossover_setup": ema_crossover.evaluate(df),
       "breakout_setup": breakout.evaluate(df),
       "rsi_divergence_setup": rsi_divergence.evaluate(df),
   }


if __name__ == "__main__":
    import json
    import sys
    sym = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE"
    print(json.dumps(analyze_symbol(sym), indent=2, default=str))
