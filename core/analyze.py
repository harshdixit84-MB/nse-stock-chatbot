"""
core/analyze.py

Live, on-demand analysis of a single NSE stock symbol.
Fetches live daily OHLCV via Angel One SmartAPI (same source your
scanner uses) and runs it through each strategy module.

Required environment variables:
  ANGEL_API_KEY, ANGEL_CLIENT_ID, ANGEL_PASSWORD, ANGEL_TOTP_SECRET
"""   
import os
os.chdir("/tmp")
import json
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
import price_action

INSTRUMENT_MASTER_URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
HISTORY_DAYS_BACK = 400  # enough calendar days to cover EMA50 + lookback windows comfortably

# Angel's own forum/docs disagree on which token returns historical
# candle data for NIFTY 50 on a given account -- "26000" is the one
# most historical getCandleData examples use successfully; "99926000"
# is the newer AMXIDX entry that works for quotes but returns empty
# candle data for some accounts. Probe both, keep whichever responds.
NIFTY_TOKEN_CANDIDATES = ["26000", "99926000"]
_index_token_cache = {"token": None}

_smart_api = None
_token_cache = {}

# Optional durable session cache (Upstash Redis REST API -- free tier,
# no credit card, plain HTTPS so no new SDK dependency). If these
# aren't set, the code below falls back to today's behavior (in-memory
# cache only, reset on every cold start). See the bottom of this file
# for what these do and why they matter.
UPSTASH_URL = os.environ.get("UPSTASH_REDIS_REST_URL", "").rstrip("/")
UPSTASH_TOKEN = os.environ.get("UPSTASH_REDIS_REST_TOKEN", "")
SESSION_CACHE_KEY = "angel_session_v1"
SESSION_TTL_SECONDS = 8 * 3600  # Angel sessions are valid for the trading day; refreshed via generateToken well before this

# NIFTY's own daily candles only change once per trading day -- caching
# this avoids adding a SECOND Angel SmartAPI historical-data call to
# EVERY single /api request (on top of the stock's own OHLCV fetch),
# which is exactly the endpoint that was already the source of prior
# rate-limit/timeout failures on this account. 6h comfortably covers
# a full trading session without serving yesterday's close once a new
# day's candle exists.
NIFTY_CACHE_KEY = "nifty_ohlcv_cache_v1"
NIFTY_CACHE_TTL_SECONDS = 6 * 3600


def _kv_command(*command_parts):
    """
    Executes a single Redis command via Upstash's REST API using the
    documented "body-style" format: POST to the bare REST URL (NOT a
    /command/key sub-path) with the entire command -- including the
    command name itself -- as a JSON array body. Confirmed against
    Upstash's own current example:
        curl https://{url} -H "Authorization: Bearer ..." \\
             -d '["SET","foo","bar","EX","60"]'
    Returns the "result" field, or None on any failure (including a
    misconfigured/unreachable KV -- this must never break analysis).
    """
    if not (UPSTASH_URL and UPSTASH_TOKEN):
        return None
    try:
        resp = requests.post(
            UPSTASH_URL,
            headers={"Authorization": f"Bearer {UPSTASH_TOKEN}"},
            json=list(command_parts),
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("error"):
            return None
        return data.get("result")
    except Exception:
        return None


def _kv_get(key):
    result = _kv_command("GET", key)
    if not result:
        return None
    try:
        return json.loads(result)
    except (TypeError, ValueError):
        return None


def _kv_set(key, value: dict, ttl_seconds: int):
    _kv_command("SET", key, json.dumps(value), "EX", str(ttl_seconds))


def _call_smartapi(fn, *args, **kwargs):
    """
    Wraps any SmartAPI SDK call (generateSession, generateToken,
    getCandleData, ...). When Angel's servers are rate-limiting this
    account, they return a plain-text body ("Access denied because of
    exceeding access rate") instead of JSON -- the SDK then throws a
    confusing internal parse error ("Couldn't parse the JSON response
    received from the server..."). This catches that specific case and
    re-raises a clear, actionable message instead. Any other exception
    passes through unchanged.
    """
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        msg = str(e)
        if "exceeding access rate" in msg.lower() or "access denied" in msg.lower():
            raise RuntimeError(
                "Angel SmartAPI is rate-limiting this account right now "
                "(too many requests in a short window). Wait a minute or "
                "two before trying again -- this isn't a bug, it's the "
                "broker's own rate limit."
            ) from e
        raise


def _fresh_login():
    "The expensive, rate-limited path: full TOTP login via generateSession."
    smart_api = SmartConnect(api_key=os.environ["ANGEL_API_KEY"])
    totp_code = pyotp.TOTP(os.environ["ANGEL_TOTP_SECRET"]).now()
    session = _call_smartapi(
        smart_api.generateSession,
        os.environ["ANGEL_CLIENT_ID"], os.environ["ANGEL_PASSWORD"], totp_code
    )
    if not session or not session.get("status"):
        raise RuntimeError(f"Angel One login failed: {session}")

    tokens = {
        "access_token": session["data"]["jwtToken"],
        "refresh_token": session["data"]["refreshToken"],
        "feed_token": session["data"]["feedToken"],
    }
    _kv_set(SESSION_CACHE_KEY, tokens, SESSION_TTL_SECONDS)
    return smart_api


def _login():
    """
    THE FIX for repeated rate-limit hits: Vercel resets all in-memory
    state (including the old _smart_api global) on every cold start,
    which on a low-traffic serverless function is often EVERY request --
    so the old code called the rate-limited TOTP login endpoint far more
    often than a human clicking "Analyze" a few times would suggest.

    Three-tier fallback, cheapest/least-rate-limited first:
      1. In-memory cache (same container, no cold start since last call)
         -- free, instant, unchanged from before.
      2. Durable cache (Upstash Redis, survives cold starts) -- reuse a
         previously issued access_token directly with ZERO Angel API
         calls. This is what actually fixes the repeated-login problem,
         but only works if UPSTASH_REDIS_REST_URL/TOKEN are configured
         (see module docstring at the bottom of this file for setup).
      3. Full TOTP login (_fresh_login) -- only when neither cache has
         a usable session. This is the one rate-limited call, now
         reserved for "truly no session anywhere" instead of "every
         cold start".

    Tier 2's cached access_token can itself expire before its TTL is
    up (Angel's JWT lifetime is shorter than what's safe to assume) --
    _fetch_ohlcv/_fetch_extended_ohlcv handle that by calling
    _refresh_or_relogin() once on an auth-looking failure, which tries
    generateToken (cheap, NOT the rate-limited endpoint) before ever
    falling back to _fresh_login again.
    """
    global _smart_api
    if _smart_api is not None:
        return _smart_api

    cached = _kv_get(SESSION_CACHE_KEY)
    if cached and cached.get("access_token"):
        smart_api = SmartConnect(
            api_key=os.environ["ANGEL_API_KEY"],
            access_token=cached["access_token"],
            refresh_token=cached.get("refresh_token"),
            feed_token=cached.get("feed_token"),
        )
        _smart_api = smart_api
        return smart_api

    _smart_api = _fresh_login()
    return _smart_api


def _refresh_or_relogin():
    """
    Called when a data call fails in a way that looks like an expired/
    invalid token. Tries the cheap refresh-token endpoint first (a
    DIFFERENT, much less strictly rate-limited endpoint than TOTP
    login); only falls back to a full fresh login if that also fails
    (e.g. the refresh_token itself is too old).
    """
    global _smart_api
    cached = _kv_get(SESSION_CACHE_KEY)
    refresh_token = cached.get("refresh_token") if cached else (
        getattr(_smart_api, "refresh_token", None) if _smart_api else None
    )

    if refresh_token and _smart_api is not None:
        try:
            resp = _call_smartapi(_smart_api.generateToken, refresh_token)
            if resp and resp.get("data", {}).get("jwtToken"):
                _kv_set(SESSION_CACHE_KEY, {
                    "access_token": resp["data"]["jwtToken"],
                    "refresh_token": refresh_token,
                    "feed_token": resp["data"].get("feedToken"),
                }, SESSION_TTL_SECONDS)
                return _smart_api
        except Exception:
            pass  # fall through to a full re-login below

    _smart_api = _fresh_login()
    return _smart_api


def _looks_like_auth_error(resp) -> bool:
    if not resp or resp.get("status"):
        return False
    message = str(resp.get("message", "")).lower()
    return any(k in message for k in ("token", "session", "unauthor", "invalid", "expired", "login"))


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
    resp = _call_smartapi(smart_api.getCandleData, params)
    time.sleep(0.35)  # respect SmartAPI's rate limit

    if _looks_like_auth_error(resp):
        # Cached access_token had expired -- try the cheap refresh path
        # (not the rate-limited TOTP login) once, then retry the call.
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


def _fetch_index_ohlcv(days_back: int = 400 + 210):
    """
    Fetch NIFTY 50 daily OHLCV for the EMA Crossover / Breakout regime
    filter (regime.py). 210 extra calendar days on top of the usual
    history window so the 200-day EMA the regime filter needs has
    enough runway even on the earliest bars analyzed.

    Cached in KV for NIFTY_CACHE_TTL_SECONDS -- without this, every
    single /api request (both "Get Plan" and "Get AI Narrative") would
    trigger a SECOND live Angel SmartAPI historical-data call in
    addition to the stock's own OHLCV fetch, on an account that's
    already rate-limit-sensitive. NIFTY's own daily candle only changes
    once per session, so there is no reason to re-fetch it per request.

    Returns None (never raises) if NIFTY data can't be fetched --
    regime.is_bullish_on() fails OPEN on None, so a NIFTY data outage
    degrades to "regime filter not applied" rather than breaking
    single-stock analysis or silently blocking every signal.
    """
    cached = _kv_get(NIFTY_CACHE_KEY)
    if cached:
        try:
            df = pd.DataFrame(cached)
            df["Date"] = pd.to_datetime(df["Date"])
            if not df.empty:
                return df
        except Exception:
            pass  # fall through to a live fetch if the cached shape is ever bad

    smart_api = _login()
    to_date = datetime.now().strftime("%Y-%m-%d %H:%M")
    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d %H:%M")

    tokens_to_try = [_index_token_cache["token"]] if _index_token_cache["token"] else NIFTY_TOKEN_CANDIDATES
    for token in tokens_to_try:
        params = {
            "exchange": "NSE",
            "symboltoken": token,
            "interval": "ONE_DAY",
            "fromdate": from_date,
            "todate": to_date,
        }
        try:
            resp = _call_smartapi(smart_api.getCandleData, params)
        except Exception:
            continue
        time.sleep(0.35)

        if resp and resp.get("status") and resp.get("data"):
            _index_token_cache["token"] = token
            df = pd.DataFrame(resp["data"], columns=["Date", "Open", "High", "Low", "Close", "Volume"])
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date").reset_index(drop=True)
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
            if not df.empty:
                # The cache read above existed but nothing ever wrote to it,
                # so every request re-fetched NIFTY live from Angel.
                cache_df = df.copy()
                cache_df["Date"] = cache_df["Date"].astype(str)
                _kv_set(NIFTY_CACHE_KEY, cache_df.to_dict(orient="records"), NIFTY_CACHE_TTL_SECONDS)
                return df

    return None


def analyze_symbol(symbol: str) -> dict:
    symbol = symbol.strip().upper()
    df = _fetch_ohlcv(symbol)
    if df is None or df.empty:
        return {"symbol": symbol, "error": "Could not fetch data. Check this is a valid NSE equity symbol."}

    last = df.iloc[-1]
    ema20 = df["Close"].ewm(span=20, adjust=False).mean().iloc[-1]
    ema50 = df["Close"].ewm(span=50, adjust=False).mean().iloc[-1]

    try:
        nifty_df = _fetch_index_ohlcv()
    except Exception:
        nifty_df = None  # regime filter fails open on None -- never block analysis on this

    return {
        "symbol": symbol,
        "last_close": round(float(last["Close"]), 2),
        "as_of": str(last["Date"].date()),
        "ema20": round(float(ema20), 2),
        "ema50": round(float(ema50), 2),
        "pullback_setup": strategy.evaluate(df, nifty_df=nifty_df),
        "crossover_setup": ema_crossover.evaluate(df, nifty_df=nifty_df),
        "breakout_setup": breakout.evaluate(df, nifty_df=nifty_df),
        "rsi_divergence_setup": rsi_divergence.evaluate(df, nifty_df=nifty_df),
        "price_action_setup": price_action.evaluate(df, nifty_df=nifty_df),
    }


if __name__ == "__main__":
    import sys
    sym = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE"
    print(json.dumps(analyze_symbol(sym), indent=2, default=str))


# ---------------------------------------------------------------------
# SETUP: durable session cache (fixes repeated Angel login rate limits)
# ---------------------------------------------------------------------
# Without this, EVERY cold start (common on low-traffic serverless --
# can be every single request) forces a fresh TOTP login, and Angel's
# login endpoint has a strict rate limit. With this, a cold start
# reuses the last issued session instead, and the login endpoint is
# only hit when there's truly no valid session anywhere.
#
# 2-minute setup, free, no credit card:
#   1. https://console.upstash.com -> Create Database (any region close
#      to your Vercel deployment's region is fine; free tier is plenty
#      for this use case).
#   2. On the database's page, copy the REST API "UPSTASH_REDIS_REST_URL"
#      and "UPSTASH_REDIS_REST_TOKEN" values.
#   3. Vercel dashboard -> this project -> Settings -> Environment
#      Variables -> add both, exactly as named above.
#   4. Redeploy. No code changes needed beyond this file -- everything
#      above already checks for these two variables and uses them
#      automatically if present, or silently falls back to today's
#      in-memory-only behavior if not.
