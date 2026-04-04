"""GET /api/customers — list for existing-customer checkout."""

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
    handler.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(204)
        _cors(self)
        self.end_headers()

    def do_GET(self):
        try:
            resp = (
                _sb()
                .table("customers")
                .select("customer_id, full_name, email, gender, birthdate")
                .eq("is_active", 1)
                .order("customer_id")
                .limit(2000)
                .execute()
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            _cors(self)
            self.end_headers()
            self.wfile.write(json.dumps({"customers": resp.data or []}).encode())
        except Exception as exc:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            _cors(self)
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(exc)}).encode())

    def log_message(self, fmt, *args):
        return
