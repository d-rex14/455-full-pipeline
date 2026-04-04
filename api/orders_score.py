"""POST /api/orders/score — batch ML scoring, persists ml_* columns."""

from http.server import BaseHTTPRequestHandler
import json
import os

from supabase import create_client

from fraud_scoring import FRAUD_THRESHOLD, score_order_id

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

_sb = None
_SCORE_CAP = 400


def _sb():
    global _sb
    if _sb is None:
        _sb = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _sb


def _cors(handler):
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(204)
        _cors(self)
        self.end_headers()

    def do_POST(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length)) if content_length else {}
        except json.JSONDecodeError:
            return self._send_json(400, {"error": "invalid JSON"})

        try:
            sb = _sb()
            order_ids = []

            if body.get("all"):
                resp = sb.table("orders").select("order_id").order("order_id", desc=True).limit(_SCORE_CAP).execute()
                order_ids = [r["order_id"] for r in (resp.data or [])]
            else:
                raw = body.get("order_ids")
                if not isinstance(raw, list) or not raw:
                    return self._send_json(400, {"error": 'send {"all": true} or {"order_ids": [1,2,...]}'})
                order_ids = [int(x) for x in raw][: _SCORE_CAP]

            results = []
            for oid in order_ids:
                scored = score_order_id(sb, oid)
                if not scored.get("ok"):
                    results.append({"order_id": oid, "ok": False, "error": scored.get("error")})
                    continue
                prob = scored["probability"]
                pred = scored["is_fraud"]
                sb.table("orders").update(
                    {
                        "ml_fraud_probability": prob,
                        "ml_predicted_fraud": pred,
                    }
                ).eq("order_id", oid).execute()
                results.append(
                    {
                        "order_id": oid,
                        "ok": True,
                        "probability": round(prob, 4),
                        "ml_predicted_fraud": pred,
                        "threshold": FRAUD_THRESHOLD,
                    }
                )

            return self._send_json(
                200,
                {"threshold": FRAUD_THRESHOLD, "scored": len(results), "results": results},
            )
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
