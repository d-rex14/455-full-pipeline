"""
Vercel serverless function: POST /api/predict
Accepts { "order_id": <int> }, queries Supabase, engineers features,
scores with the trained fraud-detection pipeline, and returns a prediction.
"""

from http.server import BaseHTTPRequestHandler
import json
import os
import joblib
import datetime

import numpy as np
import pandas as pd
from supabase import create_client

# ── Configuration ──────────────────────────────────────────────────────────────

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
FRAUD_THRESHOLD = float(os.environ.get("FRAUD_THRESHOLD", "0.428"))

MODEL_PATH    = os.path.join(os.path.dirname(__file__), "..", "model.sav")
METADATA_PATH = os.path.join(os.path.dirname(__file__), "..", "model_metadata.json")

# ── Lazy-loaded singletons ────────────────────────────────────────────────────

_model     = None
_threshold = None
_supabase  = None


def _get_supabase():
    global _supabase
    if _supabase is None:
        _supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _supabase


def _get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def _get_threshold():
    """Read the optimised threshold from model_metadata.json if present,
    otherwise fall back to the FRAUD_THRESHOLD env-var default."""
    global _threshold
    if _threshold is not None:
        return _threshold
    try:
        with open(METADATA_PATH) as f:
            _threshold = float(json.load(f).get("final_threshold", FRAUD_THRESHOLD))
    except Exception:
        _threshold = FRAUD_THRESHOLD
    return _threshold


# ── Feature engineering (mirrors Phase 3 of the notebook) ─────────────────────

def engineer_features(order, customer, shipment, items_agg):
    """
    Build a single-row DataFrame with the 32 features the model expects.

    Parameters
    ----------
    order      : dict   – row from the orders table
    customer   : dict   – row from the customers table
    shipment   : dict   – row from the shipments table (may be None)
    items_agg  : dict   – aggregated order-items stats
    """

    order_dt = pd.to_datetime(order["order_datetime"], errors="coerce")
    cust_created = pd.to_datetime(customer["created_at"], errors="coerce")
    birthdate = pd.to_datetime(customer["birthdate"], errors="coerce")

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
        # ── Categorical features ──
        "payment_method":   order.get("payment_method", "card"),
        "device_type":      order.get("device_type", "desktop"),
        "gender":           customer.get("gender", "Male"),
        "customer_segment": customer.get("customer_segment", "standard"),
        "loyalty_tier":     customer.get("loyalty_tier", "none"),
        "carrier":          shipment.get("carrier", "UPS") if shipment else "UPS",
        "shipping_method":  shipment.get("shipping_method", "standard") if shipment else "standard",
        "distance_band":    shipment.get("distance_band", "regional") if shipment else "regional",
        "ip_country":       ip_country,
        "shipping_state":   shipping_state,

        # ── Numeric features ──
        "promo_used":              int(order.get("promo_used", 0)),
        "order_subtotal":          float(order.get("order_subtotal", 0)),
        "shipping_fee":            float(order.get("shipping_fee", 0)),
        "tax_amount":              float(order.get("tax_amount", 0)),
        "risk_score":              float(order.get("risk_score", 0)),
        "customer_is_active":      int(customer.get("is_active", 1)),
        "late_delivery":           int(shipment.get("late_delivery", 0)) if shipment else 0,
        "item_count":              item_count,
        "total_items":             int(items_agg.get("total_items", 1)),
        "avg_unit_price":          float(items_agg.get("avg_unit_price", 0)),
        "unique_products":         unique_products,

        # ── Engineered features ──
        "zip_mismatch":            int(order.get("billing_zip") != order.get("shipping_zip")),
        "ip_international":        int(ip_country_raw != "US"),
        "ip_high_risk_country":    int(ip_country_raw in ("NG", "IN", "BR")),
        "is_weekend":              int(order_dt.dayofweek >= 5) if pd.notna(order_dt) else 0,
        "is_night_order":          int(order_dt.hour >= 23 or order_dt.hour <= 4) if pd.notna(order_dt) else 0,
        "order_before_account":    int(account_age_days < 0),
        "account_age_days_capped": max(account_age_days, 0),
        "customer_age_years":      customer_age_years,
        "is_high_value":           int(order_total > 500),
        "promo_high_value":        int(int(order.get("promo_used", 0)) == 1 and order_total > 500),
        "product_diversity_ratio": unique_products / max(item_count, 1),
    }

    return pd.DataFrame([row])


# ── Request handler ───────────────────────────────────────────────────────────

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length)) if content_length else {}

            order_id = body.get("order_id")
            if order_id is None:
                return self._json(400, {"error": "order_id is required"})

            sb = _get_supabase()

            # Fetch order
            order_resp = sb.table("orders").select("*").eq("order_id", order_id).execute()
            if not order_resp.data:
                return self._json(404, {"error": f"order {order_id} not found"})
            order = order_resp.data[0]

            # Fetch customer
            cust_resp = sb.table("customers").select("*").eq("customer_id", order["customer_id"]).execute()
            customer = cust_resp.data[0] if cust_resp.data else {}

            # Fetch shipment (may not exist yet for new orders)
            ship_resp = sb.table("shipments").select("*").eq("order_id", order_id).execute()
            shipment = ship_resp.data[0] if ship_resp.data else None

            # Fetch aggregated order items
            items_resp = sb.table("order_items").select("*").eq("order_id", order_id).execute()
            items = items_resp.data or []
            items_agg = {
                "item_count":      len(items),
                "total_items":     sum(i.get("quantity", 0) for i in items),
                "avg_unit_price":  np.mean([i.get("unit_price", 0) for i in items]) if items else 0,
                "unique_products": len(set(i.get("product_id") for i in items)),
            }

            # Engineer features and predict
            X = engineer_features(order, customer, shipment, items_agg)
            model     = _get_model()
            threshold = _get_threshold()
            probability = float(model.predict_proba(X)[:, 1][0])
            is_fraud    = int(probability > threshold)

            return self._json(200, {
                "order_id":    order_id,
                "is_fraud":    is_fraud,
                "probability": round(probability, 4),
                "threshold":   threshold,
            })

        except Exception as exc:
            return self._json(500, {"error": str(exc)})

    def do_GET(self):
        self._json(200, {
            "status": "ok",
            "endpoint": "POST /api/predict",
            "usage": '{"order_id": 1}',
        })

    # ── Helpers ──

    def _json(self, status, data):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
