"""POST /api/orders/score — batch ML scoring, persists ml_* columns.

ML logic is inlined so Vercel bundles a single file + model.sav (no sibling import issues).
"""

from http.server import BaseHTTPRequestHandler
import json
import os
import pickle

import numpy as np
import pandas as pd
from supabase import create_client

# ── Fraud scoring (inlined; keep in sync with predict.py / fraud_scoring.py) ──

FRAUD_THRESHOLD = float(os.environ.get("FRAUD_THRESHOLD", "0.428"))
KNOWN_IP_COUNTRIES = {"US"}
KNOWN_SHIPPING_STATES = {"CO", "OH", "MI", "TX", "NC", "AZ"}
_model = None


def _resolve_model_path():
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (
        os.path.join(here, "model.sav"),
        os.path.normpath(os.path.join(here, "..", "model.sav")),
    ):
        if os.path.isfile(p):
            return p
    return os.path.normpath(os.path.join(here, "..", "model.sav"))


def _get_model():
    global _model
    if _model is None:
        path = _resolve_model_path()
        if not os.path.isfile(path):
            raise FileNotFoundError(
                "model.sav not found. Train with fraud_detection.ipynb (serialization cell), "
                "place model.sav at the project root, commit it, and redeploy so Vercel can bundle it."
            )
        with open(path, "rb") as f:
            _model = pickle.load(f)
    return _model


def _engineer_features(order, customer, shipment, items_agg):
    order_dt = pd.to_datetime(order["order_datetime"], errors="coerce", utc=True)
    cust_created = pd.to_datetime(customer.get("created_at"), errors="coerce", utc=True)
    birthdate = pd.to_datetime(customer.get("birthdate"), errors="coerce", utc=True)

    account_age_days = (order_dt - cust_created).days if pd.notna(order_dt) and pd.notna(cust_created) else 0
    customer_age_years = (order_dt - birthdate).days // 365 if pd.notna(order_dt) and pd.notna(birthdate) else 30

    item_count = items_agg.get("item_count", 1)
    unique_products = items_agg.get("unique_products", 1)

    ip_country_raw = order.get("ip_country", "US")
    shipping_state_raw = order.get("shipping_state", "Other")
    ip_country = ip_country_raw if ip_country_raw in KNOWN_IP_COUNTRIES else "Other"
    shipping_state = shipping_state_raw if shipping_state_raw in KNOWN_SHIPPING_STATES else "Other"

    order_total = float(order.get("order_total", 0))

    row = {
        "payment_method": order.get("payment_method", "card"),
        "device_type": order.get("device_type", "desktop"),
        "gender": customer.get("gender", "Male"),
        "customer_segment": customer.get("customer_segment", "standard"),
        "loyalty_tier": customer.get("loyalty_tier", "none"),
        "carrier": shipment.get("carrier", "UPS") if shipment else "UPS",
        "shipping_method": shipment.get("shipping_method", "standard") if shipment else "standard",
        "distance_band": shipment.get("distance_band", "regional") if shipment else "regional",
        "ip_country": ip_country,
        "shipping_state": shipping_state,
        "promo_used": int(order.get("promo_used", 0)),
        "order_subtotal": float(order.get("order_subtotal", 0)),
        "shipping_fee": float(order.get("shipping_fee", 0)),
        "tax_amount": float(order.get("tax_amount", 0)),
        "risk_score": float(order.get("risk_score", 0)),
        "customer_is_active": int(customer.get("is_active", 1)),
        "late_delivery": int(shipment.get("late_delivery", 0)) if shipment else 0,
        "item_count": item_count,
        "total_items": int(items_agg.get("total_items", 1)),
        "avg_unit_price": float(items_agg.get("avg_unit_price", 0)),
        "unique_products": unique_products,
        "zip_mismatch": int(order.get("billing_zip") != order.get("shipping_zip")),
        "ip_international": int(ip_country_raw != "US"),
        "ip_high_risk_country": int(ip_country_raw in ("NG", "IN", "BR")),
        "is_weekend": int(order_dt.dayofweek >= 5) if pd.notna(order_dt) else 0,
        "is_night_order": int(order_dt.hour >= 23 or order_dt.hour <= 4) if pd.notna(order_dt) else 0,
        "order_before_account": int(account_age_days < 0),
        "account_age_days_capped": max(account_age_days, 0),
        "customer_age_years": customer_age_years,
        "is_high_value": int(order_total > 500),
        "promo_high_value": int(int(order.get("promo_used", 0)) == 1 and order_total > 500),
        "product_diversity_ratio": unique_products / max(item_count, 1),
    }

    return pd.DataFrame([row])


def _items_aggregate(items):
    items = items or []
    return {
        "item_count": len(items),
        "total_items": sum(i.get("quantity", 0) for i in items),
        "avg_unit_price": float(np.mean([i.get("unit_price", 0) for i in items])) if items else 0.0,
        "unique_products": len(set(i.get("product_id") for i in items)),
    }


def _score_order_id(sb, order_id, threshold=None):
    thr = float(threshold) if threshold is not None else FRAUD_THRESHOLD
    try:
        oid = int(order_id)
    except (TypeError, ValueError):
        return {"ok": False, "error": "invalid order_id"}

    order_resp = sb.table("orders").select("*").eq("order_id", oid).execute()
    if not order_resp.data:
        return {"ok": False, "error": f"order {oid} not found"}

    order = order_resp.data[0]
    cust_resp = sb.table("customers").select("*").eq("customer_id", order["customer_id"]).execute()
    customer = cust_resp.data[0] if cust_resp.data else {}

    ship_resp = sb.table("shipments").select("*").eq("order_id", oid).execute()
    shipment = ship_resp.data[0] if ship_resp.data else None

    items_resp = sb.table("order_items").select("*").eq("order_id", oid).execute()
    items_agg = _items_aggregate(items_resp.data)

    X = _engineer_features(order, customer, shipment, items_agg)
    model = _get_model()
    probability = float(model.predict_proba(X)[:, 1][0])
    is_fraud = int(probability > thr)

    return {
        "ok": True,
        "order_id": oid,
        "probability": round(probability, 6),
        "is_fraud": is_fraud,
        "threshold": thr,
    }


# ── HTTP handler ──────────────────────────────────────────────────────────────

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

_supabase_client = None
_SCORE_CAP = 400


def _get_supabase():
    global _supabase_client
    if _supabase_client is None:
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _supabase_client


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
            sb = _get_supabase()
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
                scored = _score_order_id(sb, oid)
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
