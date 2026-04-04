"""
Vercel Cron: GET /api/cron

Vercel invokes this on a schedule. Set CRON_SECRET in the project env; Vercel sends
Authorization: Bearer <CRON_SECRET> on cron invocations.
"""

from http.server import BaseHTTPRequestHandler
import json
import os


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        secret = (os.environ.get("CRON_SECRET") or "").strip()
        auth = self.headers.get("Authorization") or ""
        if not secret or auth != f"Bearer {secret}":
            self._send(401, "text/plain; charset=utf-8", b"Unauthorized")
            return

        # Hook: add nightly work here (e.g. Supabase maintenance, batch scoring).
        payload = {"ok": True}
        self._send(
            200,
            "application/json; charset=utf-8",
            json.dumps(payload).encode("utf-8"),
        )

    def log_message(self, fmt, *args):
        return

    def _send(self, status, content_type, body: bytes):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.end_headers()
        self.wfile.write(body)
