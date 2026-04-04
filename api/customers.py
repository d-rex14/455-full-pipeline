"""GET /api/customers — list for existing-customer checkout."""

from http.server import BaseHTTPRequestHandler
import json
import os

from supabase import create_client

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

_supabase_client = None


def _get_supabase():
    global _supabase_client
    if _supabase_client is None:
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _supabase_client


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
                _get_supabase()
                .table("customers")
                .select(
                    "customer_id, full_name, email, gender, birthdate, city, state, zip_code, "
                    "customer_segment, loyalty_tier"
                )
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
