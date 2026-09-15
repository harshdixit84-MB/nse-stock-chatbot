import sys
import os
import json
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

import requests

# Make the sibling "core" folder importable from inside "api"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "core"))

import verdict
from analyze import _kv_get, _kv_set

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

# Set to True to re-enable /analyze on Telegram. Currently off so the
# web dashboard's Analyze button (api/index.py's do_GET) is the ONLY
# thing drawing on the shared Angel SmartAPI rate-limit budget while
# it's being tested -- both surfaces use the exact same account.
TELEGRAM_ANALYZE_ENABLED = False


def _send_telegram_message(chat_id, text):
    try:
        requests.post(
            TELEGRAM_API_URL,
            json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
            timeout=15,
        )
    except Exception:
        pass  # best-effort -- don't crash the webhook handler over a failed reply


def _format_verdict_message(result: dict) -> str:
    if "error" in result:
        return f"\u26a0\ufe0f {result['error']}"

    lines = [f"*{result['symbol']}* \u2014 as of {result['as_of']}",
              f"Last close: {result['last_close']}"]

    verdict_type = result["verdict"]

    if verdict_type == "BUY":
        plan = result["trade_plan"]
        bt = result["backtest"]
        lines.append("")
        lines.append(f"\u2705 *BUY* via {result['strategy']}")
        lines.append(f"Entry: {plan['entry_price']} | Stop: {plan['stop_loss']} | Target: {plan['target']}")
        lines.append(f"Risk:Reward = 1:{plan['reward_risk_ratio']}")
        lines.append(f"Backtest: {bt['win_rate_pct']}% win rate over {bt['signals']} signals, profit factor {bt['profit_factor']}")
        if bt.get("low_sample_warning"):
            lines.append("\u26a0\ufe0f Small sample size -- treat with caution")
    elif verdict_type == "BUY_NO_TRACK_RECORD":
        plan = result["trade_plan"]
        lines.append("")
        lines.append(f"\u26a0\ufe0f *{result['strategy']}* setup active, but no historical track record on this stock")
        lines.append(f"Entry: {plan['entry_price']} | Stop: {plan['stop_loss']} | Target: {plan['target']}")
    else:
        lines.append("")
        lines.append("No active setup today.")

    macd = result.get("momentum_confirmation")
    if macd:
        lines.append(f"\nMomentum (MACD): {macd['status']}")

    chart = result.get("chart_read")
    if chart and "error" not in chart:
        lines.append(f"\n\U0001F4CA *Chart Read*")
        lines.append(f"Trend: {chart['trend']}")
        if chart["nearest_support"] is not None:
            lines.append(f"Support: {chart['nearest_support']} ({chart['distance_to_support_pct']}% below)")
        if chart["nearest_resistance"] is not None:
            lines.append(f"Resistance: {chart['nearest_resistance']} ({chart['distance_to_resistance_pct']}% above)")
        lines.append(f"Today's candle: {chart['todays_candle_pattern']} ({chart['todays_candle_bias']})")
        lines.append(f"RSI(14): {chart['rsi_14']} -- {chart['rsi_zone']}")
        lines.append(f"Volume: {chart['volume_vs_20d_avg']}")
        lines.append(f"Reversal watch: {chart['reversal_watch']}")

    lines.append("\n_Ranking by backtested win rate:_")
    for r in result["strategy_ranking"]:
        flag = "\U0001F7E2" if r["active_today"] else "\u26aa"
        wr = f"{r['win_rate_pct']}%" if r["win_rate_pct"] is not None else "n/a"
        lines.append(f"{flag} {r['strategy']}: {wr} ({r['signals']} signals)")

    return "\n".join(lines)


class handler(BaseHTTPRequestHandler):

    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        query = parse_qs(urlparse(self.path).query)
        symbol = query.get("symbol", [""])[0].strip()
        include_narrative = query.get("narrative", ["false"])[0].strip().lower() in ("1", "true", "yes")

        if not symbol:
            self.send_response(400)
            self.send_header("Content-type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Missing 'symbol' query parameter"}).encode())
            return

        try:
            result = verdict.get_verdict(symbol, include_narrative=include_narrative)
            status = 200
        except Exception as e:
            result = {"error": str(e)}
            status = 500

        self.send_response(status)
        self.send_header("Content-type", "application/json")
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps(result, default=str).encode())
        return

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length else b"{}"

        try:
            update = json.loads(body)
        except Exception:
            update = {}

        # Telegram retries a webhook delivery if it doesn't get a prompt
        # response -- and a full 5-strategy backtest over ~5 years of
        # daily data is genuinely slow, easily slow enough to trigger
        # that. Without this check, each retry re-runs the ENTIRE
        # analysis from scratch and sends ANOTHER reply -- which is
        # exactly the "3 requests sent by the bot itself" symptom for
        # one /analyze call. update_id is unique per Telegram update
        # and IDENTICAL across retries of the same delivery, so this
        # skips any update we've already started handling, using the
        # same durable KV cache _login() already relies on.
        update_id = update.get("update_id")
        if update_id is not None:
            dedupe_key = f"tg_update_{update_id}"
            if _kv_get(dedupe_key):
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"ok": true, "duplicate": true}')
                return
            _kv_set(dedupe_key, {"seen": True}, 600)  # 10 min -- comfortably longer than Telegram's retry window

        message = update.get("message", {})
        chat_id = message.get("chat", {}).get("id")
        text = (message.get("text") or "").strip()

        if chat_id:
            lowered = text.lower()
            if lowered.startswith("/analyze"):
                if not TELEGRAM_ANALYZE_ENABLED:
                    _send_telegram_message(
                        chat_id,
                        "The /analyze command is temporarily paused here while the "
                        "web dashboard's Analyze button is being tested -- both draw "
                        "from the same Angel SmartAPI account, so running both at once "
                        "makes debugging harder. Use the web dashboard for now."
                    )
                else:
                    parts = text.split(maxsplit=1)
                    symbol = parts[1].strip() if len(parts) > 1 else ""
                    if not symbol:
                        _send_telegram_message(chat_id, "Usage: /analyze SYMBOL (e.g. /analyze RELIANCE)")
                    else:
                        _send_telegram_message(chat_id, f"Analyzing {symbol.upper()}... this can take up to 30 seconds.")
                        try:
                            result = verdict.get_verdict(symbol)
                            reply = _format_verdict_message(result)
                        except Exception as e:
                            reply = f"Error analyzing {symbol}: {e}"
                        _send_telegram_message(chat_id, reply)
            elif lowered in ("/start", "/help"):
                _send_telegram_message(chat_id, "Send /analyze SYMBOL to get a live trade verdict, e.g. /analyze RELIANCE")

        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok": true}')
        return
