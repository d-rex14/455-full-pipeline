"""
Vercel serverless function: POST /api/predict
Accepts { "order_id": <int> }, queries Supabase, engineers features,
scores with the trained fraud-detection pipeline, and returns a prediction.
"""

from http.server import BaseHTTPRequestHandler
import json
import os

from supabase import create_client

from fraud_scoring import score_order_id

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

_supabase = None


def _get_supabase():
    global _supabase
    if _supabase is None:
        _supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _supabase


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length)) if content_length else {}

            order_id = body.get("order_id")
            if order_id is None:
                return self._json(400, {"error": "order_id is required"})

            sb = _get_supabase()
            result = score_order_id(sb, order_id)
            if not result.get("ok"):
                status = 404 if "not found" in result.get("error", "").lower() else 400
                return self._json(status, {"error": result.get("error", "score failed")})

            return self._json(200, {
                "order_id": result["order_id"],
                "is_fraud": result["is_fraud"],
                "probability": round(result["probability"], 4),
                "threshold": result["threshold"],
            })

        except Exception as exc:
            return self._json(500, {"error": str(exc)})

    def do_GET(self):
        self._json(200, {
            "status": "ok",
            "endpoint": "POST /api/predict",
            "usage": '{"order_id": 1}',
        })

    def _json(self, status, data):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
