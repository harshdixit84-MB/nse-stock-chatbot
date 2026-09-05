import sys
import os
import json
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# Make the sibling "core" folder importable from inside "api"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "core"))

import verdict


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
