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

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"


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

        if not symbol:
            self.send_response(400)
            self.send_header("Content-type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Missing 'symbol' query parameter"}).encode())
            return

        try:
            result = verdict.get_verdict(symbol)
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

        message = update.get("message", {})
        chat_id = message.get("chat", {}).get("id")
        text = (message.get("text") or "").strip()

        if chat_id:
            lowered = text.lower()
            if lowered.startswith("/analyze"):
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
