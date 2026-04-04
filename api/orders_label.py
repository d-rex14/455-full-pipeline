"""PATCH /api/orders/label — set is_fraud + admin_reviewed."""

from http.server import BaseHTTPRequestHandler
import json
import os

from supabase import create_client

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

_sb = None


def _sb():
    global _sb
    if _sb is None:
        _sb = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _sb


def _cors(handler):
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "PATCH, POST, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(204)
        _cors(self)
        self.end_headers()

    def do_PATCH(self):
        return self._handle()

    def do_POST(self):
        return self._handle()

    def _handle(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length)) if content_length else {}
        except json.JSONDecodeError:
            return self._send_json(400, {"error": "invalid JSON"})

        try:
            order_id = body.get("order_id")
            if order_id is None:
                return self._send_json(400, {"error": "order_id is required"})
            raw = body.get("is_fraud")
            if raw in (True, 1, "1"):
                is_fraud = 1
            elif raw in (False, 0, "0"):
                is_fraud = 0
            else:
                return self._send_json(400, {"error": "is_fraud must be 0 or 1"})

            oid = int(order_id)
            sb = _sb()
            upd = (
                sb.table("orders")
                .update({"is_fraud": int(is_fraud), "admin_reviewed": 1})
                .eq("order_id", oid)
                .select("order_id, is_fraud, admin_reviewed")
                .execute()
            )
            if not upd.data:
                return self._send_json(404, {"error": f"order {oid} not found"})

            return self._send_json(200, {"ok": True, "order": upd.data[0]})
        except Exception as exc:
            return self._send_json(500, {"error": str(exc)})

    def _send_json(self, status, data):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        _cors(self)
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, fmt, *args):
        return
