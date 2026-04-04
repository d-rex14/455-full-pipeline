"""
GET /api/orders — admin list with customer embed
POST /api/orders — place order (customer + order + items + shipment)
"""

from datetime import datetime, timezone
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


def _cors(handler, methods="GET, POST, OPTIONS"):
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", methods)
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")


def _path_clean(handler):
    return handler.path.split("?", 1)[0].rstrip("/")


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(204)
        _cors(self)
        self.end_headers()

    def do_GET(self):
        if _path_clean(self) != "/api/orders":
            return self._json(404, {"error": "not found"})
        try:
            sb = _get_supabase()
            resp = (
                sb.table("orders")
                .select(
                    "order_id, customer_id, order_datetime, order_total, order_subtotal, "
                    "shipping_fee, tax_amount, is_fraud, ml_fraud_probability, ml_predicted_fraud, "
                    "admin_reviewed, payment_method, shipping_state, ip_country"
                )
                .order("order_id", desc=True)
                .limit(500)
                .execute()
            )
            rows = resp.data or []
            cids = list({r["customer_id"] for r in rows if r.get("customer_id") is not None})
            cust_map = {}
            if cids:
                cr = sb.table("customers").select("customer_id, full_name, email").in_("customer_id", cids).execute()
                for c in cr.data or []:
                    cust_map[c["customer_id"]] = {"full_name": c.get("full_name"), "email": c.get("email")}
            for r in rows:
                info = cust_map.get(r.get("customer_id"))
                r["customers"] = info or {"full_name": None, "email": None}
            return self._json(200, {"orders": rows})
        except Exception as exc:
            return self._json(500, {"error": str(exc)})

    def do_POST(self):
        if _path_clean(self) != "/api/orders":
            return self._json(404, {"error": "not found"})
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length)) if content_length else {}
        except json.JSONDecodeError:
            return self._json(400, {"error": "invalid JSON"})

        try:
            result = _create_order(_get_supabase(), body)
            return self._json(result["status"], result["body"])
        except ValueError as ve:
            return self._json(400, {"error": str(ve)})
        except Exception as exc:
            return self._json(500, {"error": str(exc)})

    def _json(self, status, data):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        _cors(self)
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, fmt, *args):
        return


def _iso_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _create_order(sb, body):
    lines = body.get("lines") or []
    if not isinstance(lines, list) or len(lines) < 1:
        raise ValueError("lines must be a non-empty array of {product_id, quantity}")

    customer_mode = (body.get("customer_mode") or "new").lower()
    now = _iso_now()

    if customer_mode == "existing":
        cid = body.get("customer_id")
        if cid is None:
            raise ValueError("customer_id required when customer_mode is existing")
        customer_id = int(cid)
        check = sb.table("customers").select("customer_id").eq("customer_id", customer_id).execute()
        if not check.data:
            raise ValueError(f"customer {customer_id} not found")
    else:
        email = (body.get("email") or "").strip()
        full_name = (body.get("full_name") or "").strip()
        if not email or not full_name:
            raise ValueError("full_name and email required for new customer")
        gender = body.get("gender") or "Male"
        birthdate = body.get("birthdate") or "1990-01-01"
        cust_row = {
            "full_name": full_name,
            "email": email,
            "gender": gender,
            "birthdate": birthdate,
            "created_at": body.get("created_at") or now,
            "city": body.get("city"),
            "state": body.get("state"),
            "zip_code": body.get("zip_code"),
            "customer_segment": body.get("customer_segment") or "standard",
            "loyalty_tier": body.get("loyalty_tier") or "none",
            "is_active": int(body.get("is_active", 1)),
        }
        ins = sb.table("customers").insert(cust_row).select("customer_id").execute()
        if not ins.data:
            raise ValueError("failed to create customer")
        customer_id = ins.data[0]["customer_id"]

    product_rows = {}
    for line in lines:
        pid = int(line.get("product_id"))
        qty = int(line.get("quantity", 0))
        if qty < 1:
            raise ValueError("each line needs quantity >= 1")
        if pid not in product_rows:
            pr = sb.table("products").select("product_id, price").eq("product_id", pid).eq("is_active", 1).execute()
            if not pr.data:
                raise ValueError(f"product {pid} not found or inactive")
            product_rows[pid] = float(pr.data[0]["price"])

    order_subtotal = 0.0
    resolved_lines = []
    for line in lines:
        pid = int(line.get("product_id"))
        qty = int(line.get("quantity", 1))
        unit = float(line.get("unit_price")) if line.get("unit_price") is not None else product_rows[pid]
        line_total = round(unit * qty, 2)
        order_subtotal += line_total
        resolved_lines.append(
            {"product_id": pid, "quantity": qty, "unit_price": unit, "line_total": line_total}
        )

    order_subtotal = round(order_subtotal, 2)
    shipping_fee = float(body.get("shipping_fee", 9.99))
    tax_rate = float(body.get("tax_rate", 0.06))
    tax_amount = round(order_subtotal * tax_rate, 2)
    order_total = round(order_subtotal + shipping_fee + tax_amount, 2)

    risk_score = body.get("risk_score")
    if risk_score is None:
        risk_score = min(95.0, max(5.0, order_subtotal / 25.0))

    order_row = {
        "customer_id": customer_id,
        "order_datetime": body.get("order_datetime") or now,
        "billing_zip": body.get("billing_zip") or "00000",
        "shipping_zip": body.get("shipping_zip") or body.get("billing_zip") or "00000",
        "shipping_state": body.get("shipping_state") or "TX",
        "payment_method": body.get("payment_method") or "card",
        "device_type": body.get("device_type") or "desktop",
        "ip_country": body.get("ip_country") or "US",
        "promo_used": int(body.get("promo_used", 0)),
        "promo_code": body.get("promo_code"),
        "order_subtotal": order_subtotal,
        "shipping_fee": shipping_fee,
        "tax_amount": tax_amount,
        "order_total": order_total,
        "risk_score": float(risk_score),
        "is_fraud": int(body.get("is_fraud", 0)),
    }

    oins = sb.table("orders").insert(order_row).select("order_id").execute()
    if not oins.data:
        raise ValueError("failed to create order")
    order_id = oins.data[0]["order_id"]

    item_payload = [
        {
            "order_id": order_id,
            "product_id": rl["product_id"],
            "quantity": rl["quantity"],
            "unit_price": rl["unit_price"],
            "line_total": rl["line_total"],
        }
        for rl in resolved_lines
    ]
    sb.table("order_items").insert(item_payload).execute()

    ship = body.get("shipment") or {}
    order_dt = order_row["order_datetime"]
    ship_row = {
        "order_id": order_id,
        "ship_datetime": ship.get("ship_datetime") or order_dt,
        "carrier": ship.get("carrier") or "UPS",
        "shipping_method": ship.get("shipping_method") or "standard",
        "distance_band": ship.get("distance_band") or "regional",
        "promised_days": int(ship.get("promised_days", 5)),
        "actual_days": int(ship.get("actual_days", 5)),
        "late_delivery": int(ship.get("late_delivery", 0)),
    }
    sb.table("shipments").insert(ship_row).execute()

    return {
        "status": 201,
        "body": {
            "order_id": order_id,
            "customer_id": customer_id,
            "order_total": order_total,
            "order_subtotal": order_subtotal,
        },
    }
