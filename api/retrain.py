"""
Vercel serverless function: POST /api/retrain
Fetches all labeled data from Supabase, retrains the fraud-detection
pipeline using the same logic as the notebook, and uploads the new
model to Supabase Storage (bucket: "models").

Authorization: pass the RETRAIN_SECRET env var value as the
  X-Retrain-Secret header (or Bearer token).
"""

from http.server import BaseHTTPRequestHandler
import io
import json
import os
import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    f1_score, precision_recall_curve, recall_score,
    roc_auc_score, accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from supabase import create_client

# ── Config ─────────────────────────────────────────────────────────────────────
SUPABASE_URL    = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY    = os.environ.get("SUPABASE_KEY", "")
RETRAIN_SECRET  = os.environ.get("RETRAIN_SECRET", "")
STORAGE_BUCKET  = "models"
RECALL_FLOOR    = 0.75          # minimum fraud recall before threshold tuning

# Features the model is trained on (must stay in sync with engineer_features)
CATEGORICAL_FEATURES = [
    "payment_method", "device_type", "gender", "customer_segment",
    "loyalty_tier", "carrier", "shipping_method", "distance_band",
    "ip_country", "shipping_state",
]
NUMERIC_FEATURES = [
    "promo_used", "order_subtotal", "shipping_fee", "tax_amount",
    "risk_score", "customer_is_active", "late_delivery",
    "item_count", "total_items", "avg_unit_price", "unique_products",
    "zip_mismatch", "ip_international", "ip_high_risk_country",
    "order_hour", "order_day_of_week", "order_month",
    "is_weekend", "is_night_order", "order_before_account",
    "account_age_days_capped", "customer_age_years",
    "is_high_value", "promo_high_value", "product_diversity_ratio",
]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "is_fraud"


# ── Data helpers ───────────────────────────────────────────────────────────────

def _fetch_all(sb, table):
    """Page through a Supabase table and return all rows as a DataFrame."""
    rows = []
    page, page_size = 0, 1000
    while True:
        resp = (
            sb.table(table)
            .select("*")
            .range(page * page_size, (page + 1) * page_size - 1)
            .execute()
        )
        batch = resp.data or []
        rows.extend(batch)
        if len(batch) < page_size:
            break
        page += 1
    return pd.DataFrame(rows)


def fetch_and_join(sb):
    """Replicate the notebook's master SQL join using Supabase queries."""
    orders    = _fetch_all(sb, "orders")
    customers = _fetch_all(sb, "customers")
    shipments = _fetch_all(sb, "shipments")
    items     = _fetch_all(sb, "order_items")

    # Aggregate order_items per order
    if not items.empty:
        items_agg = items.groupby("order_id").agg(
            item_count    = ("order_item_id", "count"),
            total_items   = ("quantity", "sum"),
            avg_unit_price = ("unit_price", "mean"),
            unique_products = ("product_id", "nunique"),
        ).reset_index()
    else:
        items_agg = pd.DataFrame(columns=[
            "order_id", "item_count", "total_items",
            "avg_unit_price", "unique_products",
        ])

    df = (
        orders
        .merge(customers.add_suffix("_cust").rename(columns={"customer_id_cust": "customer_id"}),
               on="customer_id", how="left")
        .merge(shipments.add_suffix("_ship").rename(columns={"order_id_ship": "order_id"}),
               on="order_id", how="left")
        .merge(items_agg, on="order_id", how="left")
    )

    # Rename customer columns that were suffixed
    rename = {
        "gender_cust":            "gender",
        "birthdate_cust":         "birthdate",
        "created_at_cust":        "customer_created_at",
        "customer_segment_cust":  "customer_segment",
        "loyalty_tier_cust":      "loyalty_tier",
        "is_active_cust":         "customer_is_active",
        # shipment columns
        "carrier_ship":           "carrier",
        "shipping_method_ship":   "shipping_method",
        "distance_band_ship":     "distance_band",
        "late_delivery_ship":     "late_delivery",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    return df


# ── Feature engineering (mirrors notebook Cell 13) ────────────────────────────

def engineer_features(df):
    df = df.copy()

    df["zip_mismatch"]       = (df["billing_zip"] != df["shipping_zip"]).astype(int)
    df["ip_international"]   = (df["ip_country"] != "US").astype(int)
    df["ip_high_risk_country"] = df["ip_country"].isin(["NG", "IN", "BR"]).astype(int)

    dt      = pd.to_datetime(df["order_datetime"], errors="coerce")
    cust_dt = pd.to_datetime(df.get("customer_created_at"), errors="coerce")
    birth   = pd.to_datetime(df.get("birthdate"), errors="coerce")

    df["order_hour"]        = dt.dt.hour
    df["order_day_of_week"] = dt.dt.dayofweek
    df["order_month"]       = dt.dt.month
    df["is_weekend"]        = (dt.dt.dayofweek >= 5).astype(int)
    df["is_night_order"]    = ((dt.dt.hour >= 23) | (dt.dt.hour <= 4)).astype(int)

    account_age = (dt - cust_dt).dt.days
    df["order_before_account"]   = (account_age < 0).astype(int)
    df["account_age_days_capped"] = account_age.clip(lower=0)

    df["customer_age_years"] = ((dt - birth).dt.days // 365).fillna(30).astype(int)

    order_total = pd.to_numeric(df.get("order_total", 0), errors="coerce").fillna(0)
    df["is_high_value"]   = (order_total > 500).astype(int)
    df["promo_high_value"] = (
        (pd.to_numeric(df["promo_used"], errors="coerce").fillna(0) == 1)
        & (df["is_high_value"] == 1)
    ).astype(int)

    item_count = pd.to_numeric(df["item_count"], errors="coerce").fillna(1).clip(lower=1)
    df["product_diversity_ratio"] = (
        pd.to_numeric(df["unique_products"], errors="coerce").fillna(1) / item_count
    )

    return df


def cap_outliers_iqr(df, cols):
    """Winsorize columns at 1.5×IQR (mirrors pl.cap_outliers_iqr)."""
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        df[col] = df[col].clip(lower=q1 - 1.5 * iqr, upper=q3 + 1.5 * iqr)
    return df


# ── Model building ─────────────────────────────────────────────────────────────

def build_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale",  StandardScaler()),
            ]), NUMERIC_FEATURES),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("ohe",    OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), CATEGORICAL_FEATURES),
        ]
    )
    clf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        max_features="sqrt",
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("preprocess", preprocessor), ("model", clf)])


def optimize_threshold(model, X_val, y_val, recall_floor=RECALL_FLOOR):
    """Return threshold maximising F1 subject to Recall >= recall_floor."""
    y_prob = model.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_prob)
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    valid = recalls[:-1] >= recall_floor
    if valid.any():
        idx = f1s[:-1][valid].argmax()
        return float(thresholds[valid][idx])
    return 0.5


# ── Supabase Storage helpers ───────────────────────────────────────────────────

def upload_model(sb, model, threshold, metrics):
    """Serialize model + metadata and upsert both to Supabase Storage."""
    # Model bytes
    buf = io.BytesIO()
    joblib.dump(model, buf)
    model_bytes = buf.getvalue()

    sb.storage.from_(STORAGE_BUCKET).upload(
        path="model.sav",
        file=model_bytes,
        file_options={"contentType": "application/octet-stream", "upsert": "true"},
    )

    # Metadata
    meta = {
        "model_name":      "fraud_detection_pipeline",
        "trained_at_utc":  datetime.datetime.utcnow().isoformat(),
        "final_threshold": threshold,
        "features":        ALL_FEATURES,
        **metrics,
    }
    sb.storage.from_(STORAGE_BUCKET).upload(
        path="model_metadata.json",
        file=json.dumps(meta).encode(),
        file_options={"contentType": "application/json", "upsert": "true"},
    )
    return meta


# ── Request handler ────────────────────────────────────────────────────────────

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            # ── Auth ──────────────────────────────────────────────────────────
            if RETRAIN_SECRET:
                auth = self.headers.get("X-Retrain-Secret", "")
                bearer = self.headers.get("Authorization", "").removeprefix("Bearer ").strip()
                if auth != RETRAIN_SECRET and bearer != RETRAIN_SECRET:
                    return self._json(401, {"error": "unauthorized"})

            sb = create_client(SUPABASE_URL, SUPABASE_KEY)

            # ── Fetch & join ──────────────────────────────────────────────────
            df_raw = fetch_and_join(sb)
            if TARGET not in df_raw.columns:
                return self._json(422, {"error": f"'{TARGET}' column not found in orders table"})

            # Drop rows without a fraud label
            df_raw = df_raw.dropna(subset=[TARGET])
            if len(df_raw) < 100:
                return self._json(422, {
                    "error": f"only {len(df_raw)} labeled rows — need at least 100 to retrain"
                })

            # ── Feature engineering ───────────────────────────────────────────
            df = engineer_features(df_raw)

            numeric_to_cap = [
                "order_subtotal", "shipping_fee", "tax_amount", "order_total",
                "avg_unit_price", "account_age_days_capped", "customer_age_years",
                "product_diversity_ratio", "item_count", "total_items", "unique_products",
            ]
            df = cap_outliers_iqr(df, numeric_to_cap)

            # Keep only model columns + target
            missing = [c for c in ALL_FEATURES if c not in df.columns]
            if missing:
                return self._json(422, {"error": f"missing columns after feature engineering: {missing}"})

            X = df[ALL_FEATURES]
            y = df[TARGET].astype(int)

            # ── Train / test split (stratified, 25% test — matches notebook) ──
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42, stratify=y
            )

            # ── Train ─────────────────────────────────────────────────────────
            pipeline = build_pipeline()
            pipeline.fit(X_train, y_train)

            # ── Threshold optimization ────────────────────────────────────────
            threshold = optimize_threshold(pipeline, X_test, y_test)

            # ── Evaluate ──────────────────────────────────────────────────────
            y_prob  = pipeline.predict_proba(X_test)[:, 1]
            y_pred  = (y_prob >= threshold).astype(int)
            metrics = {
                "recall_fraud": round(recall_score(y_test, y_pred, pos_label=1, zero_division=0), 4),
                "f1_fraud":     round(f1_score(y_test, y_pred, pos_label=1, zero_division=0), 4),
                "roc_auc":      round(roc_auc_score(y_test, y_prob), 4),
                "accuracy":     round(accuracy_score(y_test, y_pred), 4),
                "num_train_rows": int(len(X_train)),
                "num_test_rows":  int(len(X_test)),
            }

            # ── Upload to Supabase Storage ────────────────────────────────────
            meta = upload_model(sb, pipeline, threshold, metrics)

            return self._json(200, {
                "status":    "retrained",
                "threshold": threshold,
                "metrics":   metrics,
                "trained_at_utc": meta["trained_at_utc"],
            })

        except Exception as exc:
            import traceback
            return self._json(500, {"error": str(exc), "trace": traceback.format_exc()})

    def do_GET(self):
        self._json(200, {
            "status":   "ok",
            "endpoint": "POST /api/retrain",
            "auth":     "X-Retrain-Secret header required",
        })

    def _json(self, status, data):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
